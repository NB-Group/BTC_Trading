import os
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from typing import Optional, Tuple, cast
from .utils import LOGGER

def create_kline_image(df: pd.DataFrame, save_dir: str = "cache") -> Optional[Tuple[str, str]]:
    """
    Create a K-line chart with technical indicators and save as an image file.
    Args:
        df: DataFrame with OHLCV data
        save_dir: Directory to save the image
    Returns:
        Tuple[image_path, data_time_range] if successful, else None
    """
    try:
        if df is None or len(df) < 100:
            LOGGER.warning(f"Not enough data (need at least 100, got {len(df) if df is not None else 0}), cannot generate K-line chart.")
            return None
        os.makedirs(save_dir, exist_ok=True)
        # --- Technical indicators ---
        ma_7 = pd.Series(df['close'].rolling(window=7).mean(), name='MA7')
        ma_25 = pd.Series(df['close'].rolling(window=25).mean(), name='MA25')
        ma_99 = pd.Series(df['close'].rolling(window=99).mean(), name='MA99')
        ma_20 = pd.Series(df['close'].rolling(window=20).mean(), name='MA20_BB')
        std_dev = df['close'].rolling(window=20).std()
        bollinger_upper = pd.Series(ma_20 + (2 * std_dev), name='BBUpper')
        bollinger_lower = pd.Series(ma_20 - (2 * std_dev), name='BBLower')
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = pd.Series(100 - (100 / (1 + rs)), name='RSI')
        if isinstance(df.index, pd.DatetimeIndex):
            start_time = df.index[0]
            end_time = df.index[-1]
            start_hour_str = start_time.strftime("%Y%m%d_%H")
            end_hour_str = end_time.strftime("%Y%m%d_%H")
            data_time_range = f"{start_hour_str}_to_{end_hour_str}_{len(df)}bars"
            LOGGER.info(f"Generating K-line chart, data range: {start_time} to {end_time} ({len(df)} bars)")
        else:
            data_time_range = f"generic_data_{len(df)}bars"
            LOGGER.info(f"Generating K-line chart for generic index data ({len(df)} bars)")
        market_colors = mpf.make_marketcolors(
            up='#00b764',
            down='#f54748',
            edge={'up':'#00b764', 'down':'#f54748'},
            wick={'up':'#00b764', 'down':'#f54748'},
            volume='inherit'
        )
        style = mpf.make_mpf_style(
            base_mpl_style='dark_background',
            marketcolors=market_colors,
            gridstyle='--',
            gridcolor='#404040'
        )
        plot_df = df.tail(48)
        plot_ma_7 = ma_7.tail(48)
        plot_ma_25 = ma_25.tail(48)
        plot_ma_99 = ma_99.tail(48)
        plot_bollinger_upper = bollinger_upper.tail(48)
        plot_bollinger_lower = bollinger_lower.tail(48)
        plot_ma_20_bb = ma_20.tail(48)
        plot_rsi = rsi.tail(48)
        add_plots = [
            mpf.make_addplot(plot_ma_7, color='#ffffff', width=1.0),    # MA7: white
            mpf.make_addplot(plot_ma_25, color='#f0b90a', width=1.0),   # MA25: yellow
            mpf.make_addplot(plot_ma_99, color='#8a2be2', width=1.0),   # MA99: purple
            mpf.make_addplot(plot_bollinger_upper, color='#00ffff', width=0.8),
            mpf.make_addplot(plot_bollinger_lower, color='#00ffff', width=0.8),
            mpf.make_addplot(plot_ma_20_bb, color='#ffa500', width=1.2, linestyle='--'),
            mpf.make_addplot(plot_rsi, panel=2, color='#ff00ff', ylabel='RSI'),
            mpf.make_addplot(pd.Series(70, index=plot_df.index), panel=2, color='#f54748', linestyle='--', width=0.7),
            mpf.make_addplot(pd.Series(30, index=plot_df.index), panel=2, color='#00b764', linestyle='--', width=0.7)
        ]
        image_filename = f"kline_chart_{data_time_range}.png"
        image_path = os.path.join(save_dir, image_filename)
        timeframe_str = "1H"
        title_str = "BTC/USDT K-line Chart"
        if isinstance(df.index, pd.DatetimeIndex) and isinstance(plot_df.index, pd.DatetimeIndex):
            start_time_str = plot_df.index[0].strftime("%m-%d %H:%M")
            end_time_str = plot_df.index[-1].strftime("%m-%d %H:%M")
            if hasattr(df.index, 'freqstr') and df.index.freqstr:
                timeframe_str = df.index.freqstr.replace('h','H')
            title_str = f'BTC/USDT {timeframe_str} K-line Chart ({start_time_str} ~ {end_time_str})'
        mpf.plot(
            plot_df,
            type='candle',
            style=style,
            addplot=add_plots,
            volume=True,
            title=title_str,
            ylabel='Price (USDT)',
            ylabel_lower='Volume',
            figsize=(16, 9),
            panel_ratios=(6, 2, 2),
            tight_layout=True,
            savefig=dict(fname=image_path, dpi=120, bbox_inches='tight')
        )
        LOGGER.success(f"K-line chart saved to: {image_path}")
        return image_path, data_time_range
    except Exception as e:
        LOGGER.error(f"Error generating K-line chart: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # Unit test
    from btc_predictor.data import get_data
    from btc_predictor.config import DATA_CONFIG
    print("--- Running K-line Plotter Standalone Test ---")
    symbol = cast(str, DATA_CONFIG['symbol'])
    timeframe = cast(str, DATA_CONFIG['timeframe'])
    test_data = get_data(symbol=symbol, timeframe=timeframe, limit=200)
    if test_data is not None:
        result = create_kline_image(test_data, save_dir="temp_test_cache")
        if result:
            image_path, data_time_range = result
            print(f"\nTest success! K-line chart saved to: {image_path}")
            print(f"Data time range: {data_time_range}")
            print("Check the temp_test_cache folder in the project root.")
        else:
            print("\nTest failed, could not generate K-line chart. Check logs for details.")
    else:
        print("\nFailed to get data, cannot test K-line chart generation.") 