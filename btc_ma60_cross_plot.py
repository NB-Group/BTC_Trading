import os
import pandas as pd
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Kline, Line, Grid
from pyecharts.globals import ThemeType

# 假设历史数据文件路径
SAVE_DIR = 'cache/ma60_cross_samples'
WINDOW = 60  # 每张图展示的K线数量（前后各30根）
NUM_SAMPLES = 20  # 扩充生成数量

os.makedirs(SAVE_DIR, exist_ok=True)

import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Kline, Line, Grid
from pyecharts.globals import ThemeType

# 使用项目内数据获取方法
from btc_predictor.data import get_data

SAVE_DIR = 'cache/ma60_cross_samples'
WINDOW = 60  # 每张图展示的K线数量（前后各30根）
NUM_SAMPLES = 20

os.makedirs(SAVE_DIR, exist_ok=True)


def load_data():
    # 获取BTC/USDT 1小时K线，指定since参数，获取更长历史数据
    df = get_data('BTC/USDT', '1h', since="2023-01-01T00:00:00Z")
    return df

def find_ma60_cross(df):
    ma60 = df['close'].rolling(window=60, min_periods=1).mean()
    cross_idx = np.where(np.diff(np.sign(df['close'] - ma60)) != 0)[0]
    # 新增：筛选孤立交叉点，窗口两侧K线与MA60距离大于阈值
    filtered = []
    last = -WINDOW
    min_side = 2  # 交叉点前后至少2根K线远离MA60
    dist_thresh = 0.002  # 距离阈值，0.2%
    for idx in cross_idx:
        if idx < min_side or idx > len(df) - min_side:
            continue
        # 检查窗口内是否只有一个交叉点
        window_start = max(0, idx - WINDOW//2)
        window_end = min(len(df), idx + WINDOW//2)
        df_win = df.iloc[window_start:window_end]
        ma60_win = ma60.iloc[window_start:window_end]
        cross_mask = np.diff(np.sign(df_win['close'] - ma60_win)) != 0
        if np.sum(cross_mask) != 1:
            continue
        # 检查两侧K线与MA60距离
        left = df['close'].iloc[idx-min_side:idx]
        left_ma = ma60.iloc[idx-min_side:idx]
        right = df['close'].iloc[idx+1:idx+1+min_side]
        right_ma = ma60.iloc[idx+1:idx+1+min_side]
        left_dist = np.abs(left - left_ma) / left_ma
        right_dist = np.abs(right - right_ma) / right_ma
        if (left_dist < dist_thresh).any() or (right_dist < dist_thresh).any():
            continue
        # 距离足够远，孤立交叉
        if idx - last < min_side:
            continue
        filtered.append(idx)
        last = idx
        if len(filtered) >= NUM_SAMPLES:
            break
    return filtered, ma60

def plot_sample(df, ma60, idx, i, only_ma60_and_cross=False):
    start = max(0, idx - WINDOW//2)
    end = min(len(df), idx + WINDOW//2)
    df_win = df.iloc[start:end]
    ma60_win = ma60.iloc[start:end]
    dates = [d.strftime('%Y-%m-%d %H:%M') for d in df_win.index]
    ohlc = df_win[['open', 'close', 'low', 'high']].values.tolist()
    cross_k_idx = idx - start

    if only_ma60_and_cross:
        # 只显示交叉点及前后各1根K线（共3根），让K线和交叉点居中且放大
        start = max(0, idx - 1)
        end = min(len(df), idx + 2)
        df_win = df.iloc[start:end]
        ma60_win = ma60.iloc[start:end]
        dates = [d.strftime('%Y-%m-%d %H:%M') for d in df_win.index]
        cross_k_idx = idx - start
        cross_k = df_win.iloc[cross_k_idx]
        cross_ohlc = [[cross_k['open'], cross_k['close'], cross_k['low'], cross_k['high']]]
        cross_date = [dates[cross_k_idx]]

        # MA60线（仅3根K线），只显示线不显示点
        line = (
            Line()
            .add_xaxis(dates)
            .add_yaxis('MA60', ma60_win.tolist(), is_smooth=True, is_symbol_show=False, linestyle_opts=opts.LineStyleOpts(width=2, color='#cb1dfc'), label_opts=opts.LabelOpts(is_show=False))
        )
        # 只画交叉点K线
        kline = (
            Kline()
            .add_xaxis(cross_date)
            .add_yaxis('Kline', cross_ohlc, itemstyle_opts=opts.ItemStyleOpts(color="#FFD700", color0="#FFD700", border_color="#FFD700", border_color0="#FFD700"))
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(is_scale=True),
                yaxis_opts=opts.AxisOpts(is_scale=True),
                title_opts=opts.TitleOpts(title=f'MA60 Cross Special (Only Cross)', pos_left='center'),
                legend_opts=opts.LegendOpts(pos_top='3%'),
            )
        )
        # 只在交叉点高亮一个点
        mark_line = (
            Line()
            .add_xaxis(cross_date)
            .add_yaxis('Cross', [cross_k['close']], symbol='circle', symbol_size=36, is_symbol_show=True, linestyle_opts=opts.LineStyleOpts(width=0), itemstyle_opts=opts.ItemStyleOpts(color='red'), label_opts=opts.LabelOpts(is_show=False))
        )
        line.overlap(kline)
        line.overlap(mark_line)
        grid = (
            Grid(init_opts=opts.InitOpts(width='3840px', height='2160px', theme=ThemeType.DARK))
            .add(line, grid_opts=opts.GridOpts(pos_left='5%', pos_right='5%', height='90%'))
        )
        html_path = os.path.join(SAVE_DIR, f'ma60_cross_special_{i+1}.html')
        grid.render(html_path)
        return

    # Pyecharts画完整图
    kline = (
        Kline()
        .add_xaxis(dates)
        .add_yaxis('Kline', ohlc)
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(is_scale=True),
            yaxis_opts=opts.AxisOpts(is_scale=True),
            title_opts=opts.TitleOpts(title=f'MA60 Cross Sample #{i+1}', pos_left='center'),
            legend_opts=opts.LegendOpts(pos_top='3%'),
        )
    )
    line = (
        Line()
        .add_xaxis(dates)
        .add_yaxis('MA60', ma60_win.tolist(), is_smooth=True, is_symbol_show=False, linestyle_opts=opts.LineStyleOpts(width=2, color='#cb1dfc'), label_opts=opts.LabelOpts(is_show=False))
    )
    kline.overlap(line)
    # 标记交叉点
    cross_price = df_win.iloc[cross_k_idx]['close']
    cross_time = dates[cross_k_idx]
    mark_line = (
        Line()
        .add_xaxis([cross_time])
        .add_yaxis('Cross', [cross_price], symbol='circle', symbol_size=12, linestyle_opts=opts.LineStyleOpts(width=0), itemstyle_opts=opts.ItemStyleOpts(color='red'))
    )
    kline.overlap(mark_line)
    grid = (
        Grid(init_opts=opts.InitOpts(width='3840px', height='2160px', theme=ThemeType.DARK))
        .add(kline, grid_opts=opts.GridOpts(pos_left='5%', pos_right='5%', height='90%'))
    )
    html_path = os.path.join(SAVE_DIR, f'ma60_cross_{i+1}.html')
    grid.render(html_path)
    # 可选：用Playwright截图
    # ...如需自动截图可复用原有_kline_screenshot逻辑...

if __name__ == '__main__':
    df = load_data()
    cross_indices, ma60 = find_ma60_cross(df)
    if not cross_indices:
        print('未找到符合条件的MA60交叉样本，请增加数据量或放宽筛选参数。')
    else:
        from playwright.sync_api import sync_playwright
        from PIL import Image
        # 先画普通样本
        html_files = []
        cross_points = []
        for i, idx in enumerate(cross_indices):
            if i == 0:
                first_idx = idx
                plot_sample(df, ma60, idx, i)
                html_files.append(os.path.join(SAVE_DIR, f'ma60_cross_{i+1}.html'))
                cross_points.append((i, idx))
            elif i < NUM_SAMPLES - 1:
                plot_sample(df, ma60, idx, i)
                html_files.append(os.path.join(SAVE_DIR, f'ma60_cross_{i+1}.html'))
                cross_points.append((i, idx))
        # 特殊图用第一张样本窗口，只留交叉点K线
        plot_sample(df, ma60, first_idx, NUM_SAMPLES - 1, only_ma60_and_cross=True)
        html_files.append(os.path.join(SAVE_DIR, f'ma60_cross_special_{NUM_SAMPLES}.html'))
        cross_points.append((NUM_SAMPLES-1, first_idx))
        print(f'已生成 {min(len(cross_indices), NUM_SAMPLES)} 张MA60交叉样本图，保存在 {SAVE_DIR} 下。')

        # Playwright导出canvas为PNG
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            for i, html_path in enumerate(html_files):
                page.goto(f'file://{os.path.abspath(html_path)}', wait_until='load', timeout=60000)
                # 等待canvas加载
                page.wait_for_selector('canvas', timeout=30000)
                # 导出canvas为PNG
                png_path = html_path.replace('.html', '.png')
                # JS直接下载canvas
                page.evaluate('''() => {
                    const canvas = document.querySelector('canvas');
                    const link = document.createElement('a');
                    link.download = 'canvas.png';
                    link.href = canvas.toDataURL('image/png');
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                }''')
                # 等待文件下载
                import time
                time.sleep(2)
                # 手动保存canvas为PNG（兼容性处理）
                img_bytes = page.evaluate('''() => {
                    const canvas = document.querySelector('canvas');
                    return canvas.toDataURL('image/png');
                }''')
                import base64
                img_data = img_bytes.split(',')[1]
                with open(png_path, 'wb') as f:
                    f.write(base64.b64decode(img_data))
                print(f'已导出PNG: {png_path}')
            browser.close()

        # 已移除裁切逻辑，保留原始PNG
