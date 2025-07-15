import os
import pandas as pd
import numpy as np
import asyncio
from typing import Optional, Tuple, cast
from playwright.sync_api import sync_playwright
from pyecharts import options as opts
from pyecharts.charts import Kline, Line, Bar, Grid
from pyecharts.globals import ThemeType
from .utils import LOGGER
import time

# 确保 Playwright 浏览器已安装
def ensure_playwright_browsers_installed():
    """检查并提示安装Playwright所需浏览器"""
    import subprocess
    try:
        # 尝试静默运行，检查是否已安装
        subprocess.check_output(['playwright', 'install', '--with-deps', 'chromium'], stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        LOGGER.warning("Playwright 浏览器似乎未安装。请在终端运行 'playwright install --with-deps chromium'。")
        # 抛出异常以停止执行，因为后续步骤会失败
        raise RuntimeError("Playwright 浏览器依赖缺失，无法生成图表。") from e

# 运行一次检查
try:
    ensure_playwright_browsers_installed()
except RuntimeError as e:
    LOGGER.error(e)
    # 如果在模块加载时就失败，后续调用将无法工作。
    # 这里可以根据需要决定是退出程序还是仅仅记录错误。
    # 对于一个库来说，仅仅记录错误可能更合适。


def _create_kline_screenshot(html_path: str, image_path: str) -> None:
    """使用Playwright同步API对HTML文件中的图表元素进行截图"""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            # 使用 file:// 协议打开本地HTML文件
            page.goto(f"file://{os.path.abspath(html_path)}")
            # 等待canvas渲染出来
            page.wait_for_selector(".chart-container canvas", state='visible', timeout=15000)
            time.sleep(1)  # 等待加载动画完成
            chart_element = page.query_selector(".chart-container")
            if chart_element:
                chart_element.screenshot(path=image_path)
                LOGGER.success(f"K线图截图已保存到: {image_path}")
            else:
                LOGGER.error("无法找到 .chart-container 元素进行截图。")
        except Exception as e:
            LOGGER.error(f"Playwright截图时出错: {e}")
            try:
                html_content = page.content()
                debug_html_path = os.path.join(os.path.dirname(image_path), "debug_kline_page.html")
                with open(debug_html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                LOGGER.info(f"已将调试用的HTML页面保存到: {debug_html_path}")
            except Exception as debug_e:
                LOGGER.error(f"保存调试HTML失败: {debug_e}")
        finally:
            browser.close()

def create_kline_image(df: pd.DataFrame, save_dir: str = "cache") -> Optional[Tuple[str, str]]:
    """
    使用Pyecharts和Playwright创建专业风格的K线图。
    """
    try:
        if df is None or len(df) < 2: # 至少需要2个数据点来画线
            LOGGER.warning(f"数据不足 (需要至少2条，实际获取到 {len(df) if df is not None else 0} 条数据)，无法生成K线图。")
            return None
        os.makedirs(save_dir, exist_ok=True)

        # --- 步骤 1: 数据准备和指标计算 ---
        # 确保索引是 DatetimeIndex 并补齐缺失数据
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
        df_filled = df.reindex(full_index)
        
        # 使用前一个有效收盘价填充OHLC，成交量填0
        df_filled['close'] = df_filled['close'].ffill()
        df_filled['open'] = df_filled['open'].fillna(df_filled['close'])
        df_filled['high'] = df_filled['high'].fillna(df_filled['close'])
        df_filled['low'] = df_filled['low'].fillna(df_filled['close'])
        df_filled['volume'] = df_filled['volume'].fillna(0)
        
        # 计算技术指标
        ma5 = df_filled['close'].rolling(window=5).mean().round(2)
        ma10 = df_filled['close'].rolling(window=10).mean().round(2)
        ma20 = df_filled['close'].rolling(window=20).mean().round(2)
        ma60 = df_filled['close'].rolling(window=60).mean().round(2)
        
        std_dev = df_filled['close'].rolling(window=20).std()
        boll_upper = (ma20 + 2 * std_dev).round(2)
        boll_lower = (ma20 - 2 * std_dev).round(2)
        
        delta = df_filled['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        # 使用 np.where 避免直接在 Series 上 .replace
        safe_avg_loss = pd.Series(np.where(avg_loss == 0, 1e-9, avg_loss), index=df_filled.index)
        rs = avg_gain / safe_avg_loss
        rsi_series = pd.Series(100 - (100 / (1 + rs)), index=df_filled.index)
        rsi = rsi_series.fillna(50).round(2)

        # 准备Pyecharts所需的数据格式
        ohlc_data = df_filled[['open', 'close', 'low', 'high']].values.tolist()
        # 使用列表推导式来确保 strftime 在每个 Timestamp 上调用
        dates = [d.strftime('%Y-%m-%d %H:%M:%S') for d in df_filled.index]
        volumes = df_filled['volume'].values.tolist()

        # --- 步骤 2: 使用Pyecharts创建图表 ---
        # K线主图
        kline = (
            Kline()
            .add_xaxis(dates)
            .add_yaxis(
                "Kline",
                ohlc_data,
                itemstyle_opts=opts.ItemStyleOpts(
                    color="#26A69A",  # 阳线颜色
                    color0="#EF5350", # 阴线颜色
                    border_color="#26A69A",
                    border_color0="#EF5350",
                ),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(is_scale=True, axislabel_opts=opts.LabelOpts(is_show=False)),
                yaxis_opts=opts.AxisOpts(
                    is_scale=True,
                    splitarea_opts=opts.SplitAreaOpts(
                        is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                    ),
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(is_show=False, type_="inside", xaxis_index=[0, 1, 2], range_start=50, range_end=100),
                    opts.DataZoomOpts(is_show=True, type_="slider", xaxis_index=[0, 1, 2], pos_bottom="0%", range_start=50, range_end=100),
                ],
                title_opts=opts.TitleOpts(title="BTC/USDT 1H Chart", pos_left="center"),
                legend_opts=opts.LegendOpts(pos_top="3%"),
            )
        )

        # 均线图
        line = (
            Line()
            .add_xaxis(dates)
            .add_yaxis("MA5", ma5.tolist(), is_smooth=True, is_symbol_show=False, linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.8))
            .add_yaxis("MA10", ma10.tolist(), is_smooth=True, is_symbol_show=False, linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.8))
            .add_yaxis("MA20", ma20.tolist(), is_smooth=True, is_symbol_show=False, linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.8))
            .add_yaxis("MA60", ma60.tolist(), is_smooth=True, is_symbol_show=False, linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.8))
            .set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
        )
        
        # 布林带
        boll_line = (
            Line()
            .add_xaxis(dates)
            .add_yaxis("Upper", boll_upper.tolist(), is_smooth=True, is_symbol_show=False, linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.5, color="#808080"))
            .add_yaxis("Lower", boll_lower.tolist(), is_smooth=True, is_symbol_show=False, linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.5, color="#808080"))
            .set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
        )
        kline.overlap(line)
        kline.overlap(boll_line)

        # 成交量图
        bar = (
            Bar()
            .add_xaxis(dates)
            .add_yaxis("Volume", volumes, xaxis_index=1, yaxis_index=1, label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    is_scale=True,
                    grid_index=1,
                    boundary_gap=False,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                yaxis_opts=opts.AxisOpts(
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )

        # RSI 图
        rsi_line = (
            Line()
            .add_xaxis(dates)
            .add_yaxis("RSI", rsi.tolist(), is_smooth=True, is_symbol_show=False, xaxis_index=2, yaxis_index=2,
                       linestyle_opts=opts.LineStyleOpts(width=2, color="#cb1dfc"))
            .set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
        )

        # 组合所有图表
        grid_chart = (
            Grid(init_opts=opts.InitOpts(
                width="1600px", 
                height="900px", 
                theme=ThemeType.DARK, 
                bg_color="#100C2A"
                # 不再需要设置 chart_id，因为它不起作用
            ))
            .add(kline, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", height="60%"))
            .add(bar, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="70%", height="10%"))
            .add(rsi_line, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="80%", height="10%"))
        )

        # --- 步骤 3: 渲染HTML并截图 ---
        if isinstance(df.index, pd.DatetimeIndex):
            data_time_range = f"{df.index[0].strftime('%Y%m%d_%H')}_to_{df.index[-1].strftime('%Y%m%d_%H')}_{len(df)}bars"
        else:
            data_time_range = f"generic_data_{len(df)}bars"
        
        image_filename = f"kline_chart_{data_time_range}.png"
        image_path = os.path.join(save_dir, image_filename)
        html_filename = f"temp_kline_{data_time_range}.html"
        html_path = os.path.join(save_dir, html_filename)

        # ECharts 渲染的HTML
        grid_chart.render(path=html_path) 
        
        # 运行同步截图
        _create_kline_screenshot(html_path, image_path)
        
        # 清理临时HTML文件
        os.remove(html_path)

        return image_path, data_time_range
    
    except Exception as e:
        LOGGER.error(f"使用Pyecharts生成K线图时出错: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    from btc_predictor.data import get_data
    from btc_predictor.config import DATA_CONFIG
    print("--- 运行 Pyecharts K-line Plotter 独立测试 ---")
    symbol = cast(str, DATA_CONFIG['symbol'])
    timeframe = cast(str, DATA_CONFIG['timeframe'])
    test_data = get_data(symbol=symbol, timeframe=timeframe, limit=200)
    if test_data is not None:
        result = create_kline_image(test_data, save_dir="temp_test_cache")
        if result:
            image_path, data_time_range = result
            print(f"\n测试成功！K线图已保存到: {image_path}")
            print(f"数据时间范围: {data_time_range}")
            print("请检查项目根目录下的 temp_test_cache 文件夹。")
        else:
            print("\n测试失败，无法生成K线图。请检查日志获取详细信息。")
    else:
        print("\n获取数据失败，无法测试K线图生成。") 