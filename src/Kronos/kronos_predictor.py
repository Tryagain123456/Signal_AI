import sys
import torch
import numpy as np
import pandas as pd
import logging
import akshare as ak
from datetime import datetime, timedelta
import pandas as pd
import os
import os
from dotenv import load_dotenv
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import math
from typing import Dict, Any, Tuple

# -----------------------------------------------------------------
# 1. 日志配置 (全局)
# -----------------------------------------------------------------
log_dir = "./logs"
output_dir = "./output_images_kronos"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "kronos_prediction.log"), mode='a'),
        logging.StreamHandler()          # <-- 新增：实时打印到终端
    ]
)
logger = logging.getLogger("kronos_predictor")


# -----------------------------------------------------------------
# 2. 模型加载 (全局单例)
# -----------------------------------------------------------------
predictor = None
def _lazy_load_predictor():
    """懒加载：第一次调用时再导入模型，避免工作目录问题"""
    global predictor
    if predictor is not None:               # 已加载直接返回
        return

    # 把模型所在目录插到 sys.path，保证 from model import ... 能找到
    kronos_dir = os.path.dirname(os.path.abspath(__file__))
    if kronos_dir not in sys.path:
        sys.path.insert(0, kronos_dir)

    logger.info("开始加载模型...")
    try:
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        env_path = os.path.join(project_root, ".env")
        load_dotenv(dotenv_path=env_path)

        model_path = os.environ.get("KRONOS_MODEL_PATH")
        print(f"{model_path}\n")
        tokenizer_path = os.environ.get("KRONOS_TOKENIZER_PATH") + "/"
        print(tokenizer_path)
        from model import Kronos, KronosTokenizer, KronosPredictor

        tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
        model = Kronos.from_pretrained(model_path)
        device = "cuda:2" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        predictor = KronosPredictor(model, tokenizer, device=device)
        logger.info(f"模型加载成功，使用设备: {device}")

    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        print(f"[Kronos] 加载失败: {e}")
        predictor = None


# -----------------------------------------------------------------
# 3. 数据获取函数
# -----------------------------------------------------------------
def get_and_process_data(symbol, start_date, end_date, adjust=""):
    """获取并处理数据，包括重命名列等操作"""
    try:
        # (注意) start_date 和 end_date 期望是 datetime 或 Timestamp 对象
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            adjust=adjust
        )

        if df is None or df.empty:
            logger.warning(f"未能获取到 {symbol} 的数据。")
            return pd.DataFrame()

        df = df.rename(columns={
            "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
            "收盘": "close", "成交量": "volume", "成交额": "amount", "振幅": "amplitude",
            "涨跌幅": "pct_change", "涨跌额": "change_amount", "换手率": "turnover"
        })

        df["date"] = pd.to_datetime(df["date"])
        
        keep_cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount']
        df = df[keep_cols].copy() 
        
        float_cols = ['open', 'close', 'high', 'low', 'volume', 'amount']
        df[float_cols] = df[float_cols].astype(float)
        
        return df

    except Exception as e:
        logger.error(f"获取数据时出错 (symbol={symbol}): {e}")
        return pd.DataFrame()


# -----------------------------------------------------------------
# 4. 数据清理函数 
# -----------------------------------------------------------------
def clear_data(df: pd.DataFrame) -> pd.DataFrame:
    """高级数据清理算法 - 3σ原则 + OHLC逻辑修复 + Log变换"""
    try:
        if df.empty:
            return df
            
        original_count = len(df)
        logger.info(f"开始数据清理 (原始 {original_count} 条)...")
        
        # 1. 异常值检测和移除 (3σ原则)
        for col in ['open', 'high', 'low', 'close']:
            mean_val = df[col].mean()
            std_val = df[col].std()
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            
            before_count = len(df)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            removed_count = before_count - len(df)
            
            if removed_count > 0:
                logger.info(f"  - {col} 列移除 {removed_count} 个异常值")
        
        # 2. OHLC数据逻辑关系修复
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        logger.info("  - OHLC 逻辑关系已修复")
        
        # 3. 成交量数据处理
        if 'volume' in df.columns:
            before_count = len(df)
            df = df[df['volume'] > 0]
            zero_volume_removed = before_count - len(df)
            
            volume_q99 = df['volume'].quantile(0.99)
            before_count = len(df)
            df = df[df['volume'] <= volume_q99]
            extreme_volume_removed = before_count - len(df)
            
            logger.info(f"  - 移除零成交量: {zero_volume_removed} 条")
            logger.info(f"  - 移除极端成交量: {extreme_volume_removed} 条")

            # (关键新增) 4. 对成交量和成交额进行 log1p 变换 (控制量纲)
            # 这可以 "控制" 极端大值，使其分布更平滑
            df['volume'] = np.log1p(df['volume'])
            if 'amount' in df.columns:
                 df['amount'] = np.log1p(df['amount'])
            logger.info("  - (关键) 已对 'volume' 和 'amount' 应用 log1p 变换")

        # 5. 数据清理总结
        cleaned_count = len(df)
        removed_total = original_count - cleaned_count
        removal_rate = (removed_total / original_count * 100) if original_count > 0 else 0
        
        logger.info(f"数据清理完成: 原始 {original_count} 条 → 清理后 {cleaned_count} 条 (移除 {removal_rate:.2f}%)")
        
        return df.reset_index(drop=True)
        
    except Exception as e:
        logger.error(f"数据清理出错: {e}")
        return df.reset_index(drop=True)




# -----------------------------------------------------------------
# 5. 绘图函数 
# -----------------------------------------------------------------
def build_prediction_figure(history_data, predict_data, lookback=400):
    """创建history_data与predict_data的专业对比图"""
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('📈 价格预测对比', '📊 成交量预测对比'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    #  复制数据，并反转历史数据的 log 变换
    hist_data = history_data.tail(lookback).copy()
    
    if 'volume' in hist_data.columns:
        # 检查数据是否真的被log过 (例如，最大值 < 30)
        # log1p(100000) approx 11.5
        if hist_data['volume'].max() < 30: 
            hist_data['volume'] = np.expm1(hist_data['volume'])
            logger.info("  - (绘图) 已反转 history_data 'volume' 的 log 变换，用于绘图")

    # 历史K线图 (第一行)
    fig.add_trace(
        go.Candlestick(
            x=hist_data['timestamps'],
            open=hist_data['open'],
            high=hist_data['high'],
            low=hist_data['low'],
            close=hist_data['close'],
            name="📊 历史K线",
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            opacity=0.8
        ),
        row=1, col=1
    )
    
    # 预测K线图 (第一行)
    if predict_data is not None and not predict_data.empty:
        pred_times = predict_data.index
        
        fig.add_trace(
            go.Candlestick(
                x=pred_times,
                open=predict_data['open'],
                high=predict_data['high'],
                low=predict_data['low'],
                close=predict_data['close'],
                name="🔮 AI预测K线",
                increasing_line_color='#66bb6a',
                decreasing_line_color='#ff7043',
                opacity=0.9
            ),
            row=1, col=1
        )
        
        if not hist_data.empty:
            connection_x = [hist_data['timestamps'].iloc[-1], pred_times[0]]
            connection_y = [hist_data['close'].iloc[-1], predict_data['open'].iloc[0]]
            
            fig.add_trace(
                go.Scatter(
                    x=connection_x, y=connection_y, mode='lines',
                    name='预测连接', line=dict(color='red', width=2, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # 历史成交量 (第二行)
    if 'volume' in hist_data.columns:
        fig.add_trace(
            go.Bar(
                x=hist_data['timestamps'],
                y=hist_data['volume'], # (关键) 这是反转后的 volume
                name="📊 历史成交量",
                marker_color='lightblue',
                opacity=0.7,
                hovertemplate='时间: %{x}<br>成交量: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # 预测成交量 (第二行)
    if predict_data is not None and 'volume' in predict_data.columns:
        fig.add_trace(
            go.Bar(
                x=pred_times, # type: ignore
                y=predict_data['volume'], # (关键) 这是反转后的 volume
                name="🔮 预测成交量",
                marker_color='orange',
                opacity=0.7,
                hovertemplate='时间: %{x}<br>预测成交量: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # 布局配置
    title_end_date = "..."
    if predict_data is not None and not predict_data.empty:
        title_end_date = pred_times[-1].date() # type: ignore
        
    fig.update_layout(
        title=f"🎯 Kronos AI股票预测结果分析 ({hist_data['date'].iloc[0].date()} - {title_end_date})",
        template="plotly_white",
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    for row in [1, 2]:
        fig.update_xaxes(
            type='date', showgrid=True, gridcolor='lightgray',
            row=row, col=1
        )
    
    fig.update_xaxes(title_text="时间", row=2, col=1)
    fig.update_yaxes(title_text="价格 (¥)", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    
    return fig






# -----------------------------------------------------------------
# 6. 主预测函数 
# -----------------------------------------------------------------
from typing import Optional

# def kronos_predict(symbol: str, start_date_str: Optional[str] = None, end_date_str: Optional[str] = None, pred_len: int = 90, T: float = 1.0, top_p: float = 0.9, sample_count: int = 1) :
#     """
#     主预测函数：获取、清理、预测并绘图。
#
#     (新增) 默认使用近3年数据预测未来90天。
#     如果 `start_date_str` 或 `end_date_str` 为 None，将启用默认逻辑。
#
#     (新增) 新增参数 T, top_p, sample_count 用于控制模型预测行为。
#     """
#     _lazy_load_predictor()
#     if predictor is None:
#         logger.error("预测器未初始化，无法执行预测。")
#         return None, None
#     # --- 默认日期逻辑处理 ---
#     if end_date_str is None:
#         end_ts = datetime.now()
#         end_date_str_log = end_ts.strftime("%Y-%m-%d")
#         logger.info(f"未指定 end_date_str，使用默认值 (今天): {end_date_str_log}")
#     else:
#         end_ts = pd.Timestamp(end_date_str)
#         end_date_str_log = end_date_str
#
#     if start_date_str is None:
#         start_ts = end_ts - timedelta(days=3*365)
#         start_date_str_log = start_ts.strftime("%Y-%m-%d")
#         logger.info(f"未指定 start_date_str，使用默认值 (3年前): {start_date_str_log}")
#     else:
#         start_ts = pd.Timestamp(start_date_str)
#         start_date_str_log = start_date_str
#     # --- (修改结束) ---
#
#     if predictor is None:
#         logger.error("预测器未初始化，无法执行预测。")
#         return None, None
#
#     logger.info(f"--- [START] 开始为 {symbol} 执行预测 (周期: {start_date_str_log} to {end_date_str_log}, 预测 {pred_len} 天) ---")
#
#     # 1. 数据获取
#     df_raw = get_and_process_data(symbol, start_ts, end_ts)
#
#     if df_raw.empty:
#         logger.error(f"获取数据失败 {symbol}。")
#         return None, None
#
#     # 2. 数据清理
#     df_cleaned = clear_data(df_raw.copy())
#
#     if df_cleaned.empty or len(df_cleaned) < 10:
#         logger.error(f"清理后数据不足10条 {symbol} (剩余 {len(df_cleaned)} 条)，无法预测。")
#         return None, None
#
#
#     # 3. 在清理数据 *之后* 创建时间戳
#     #  我们必须使用 'date' 列中的 *实际交易日* 作为时间戳，
#     df_cleaned["timestamps"] = df_cleaned["date"]
#
#     # 确保 'timestamps' 列是 pd.Timestamp 类型
#     df_cleaned["timestamps"] = pd.to_datetime(df_cleaned["timestamps"])
#
#     logger.info(f"已使用 'date' 列中的实际交易日作为 'timestamps'。")
#
#
#     # 4. 准备预测输入
#     #  现在 last_timestamp 是 *真正* 的最后一个交易日
#     last_timestamp = df_cleaned["timestamps"].iloc[-1]
#     logger.info(f"历史数据中的最后一个 'timestamp' 是: {last_timestamp.date()}")
#
#     x_timestamp = pd.Series(df_cleaned["timestamps"])
#
#     #  预测的时间戳 'y_timestamp' 应该跳过周末。
#     # 我们使用 'freq="B"' (Business Day) 来生成未来的 *交易日*。
#     y_timestamp = pd.Series(
#         pd.date_range(
#             start=last_timestamp + pd.Timedelta(days=1), # 从最后历史日期的 *第二天* 开始
#             periods=pred_len,
#             freq="B"  # <--- 使用 "B" (Business Day, 周一至周五)
#         )
#     )
#
#
#     logger.info(f"预测将从 {y_timestamp.iloc[0].date()} 开始 (使用 'B' 频率跳过周末)。")
#
#     # 显式选择模型需要的列
#     model_input_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamps']
#     df_for_model = df_cleaned[model_input_cols]
#
#     logger.info(f"准备调用预测器... 历史数据 {len(df_for_model)} 条 (Volume 已 Log 变换)。")
#
#     # 5. 调用 predict
#     try:
#         with torch.inference_mode():
#             pred_df = predictor.predict(
#                 df=df_for_model,
#                 x_timestamp=x_timestamp,
#                 y_timestamp=y_timestamp,
#                 pred_len=pred_len,
#                 T=T,
#                 top_p=top_p,
#                 sample_count=sample_count
#             )
#
#         # 反转成交量的 log1p 变换
#         if 'volume' in pred_df.columns:
#             pred_df['volume'] = np.expm1(pred_df['volume'])
#             logger.info("  - (关键) 已反转预测结果 'volume' 的 log1p 变换 (expm1)")
#
#         logger.info("预测成功。")
#         print("\n--- 预测结果 (Head) ---")
#         print(pred_df[["open", "high", "low", "close", "volume"]].head())
#         print("------------------------\n")
#
#         # 6. 绘图
#         logger.info("生成预测对比图...")
#         fig = build_prediction_figure(df_cleaned, pred_df)
#
#         #  保存图表到本地文件 ---
#         try:
#             global output_dir
#
#             #  safe_end_date_str 现在基于 end_ts (即 'today' 或指定的结束日期)
#             safe_end_date_str = end_ts.strftime("%Y%m%d")
#             filename = f"{symbol}_{safe_end_date_str}_pred_{pred_len}d.html"
#             save_path = os.path.join(output_dir, filename)
#
#             fig.write_html(save_path)
#
#             logger.info(f"预测图表已成功保存到: {save_path}")
#             print(f"f 预测图表已成功保存到: {save_path}")
#
#         except Exception as e:
#             logger.error(f"保存 HTML 图表失败: {e}")
#             print(f"❌ 保存 HTML 图表失败: {e}")
#
#
#         logger.info(f"--- [END] 预测完成 {symbol} ---")
#         return df_cleaned, pred_df
#
#     except Exception as e:
#         logger.exception(f"预测或绘图过程中发生严重错误: {e}")
#         return df_cleaned, None
#





# 常量：正态分位点（单尾）
Z_95 = -1.645
Z_99 = -2.33

def analyze_prediction_df(pred_df: pd.DataFrame, freq_per_year: int = 252, bootstrap_iters: int = 1000) -> Dict[str, Any]:
    """
    输入:
      pred_df: 预测结果 DataFrame，索引为 pd.DatetimeIndex，必须包含 'close' 列（可含 volume）
      freq_per_year: 年交易日数（股票通常用252）
      bootstrap_iters: 若只有单一路径，bootstrap 用于估计不确定性

    返回:
      metrics: 字典，包含统计量、风险指标、趋势判断、仓位建议、报告文本等
    """
    result: Dict[str, Any] = {}
    df = pred_df.copy().sort_index()

    # 1) 基本检查
    if 'close' not in df.columns:
        raise ValueError("pred_df 必须包含 'close' 列")

    # 2) 计算对数收益（相对变化）
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df = df.dropna(subset=['log_ret'])
    n = len(df)
    if n == 0:
        raise ValueError("预测数据长度不足以计算收益")

    # 3) 基本统计
    mean_daily = df['log_ret'].mean()
    std_daily = df['log_ret'].std(ddof=1)
    sharpe_daily = mean_daily / (std_daily + 1e-12)

    annualized_return = (np.exp(mean_daily * freq_per_year) - 1)
    annualized_vol = std_daily * np.sqrt(freq_per_year)

    result['n_days'] = int(n)
    result['mean_daily'] = float(mean_daily)
    result['std_daily'] = float(std_daily)
    result['annualized_return'] = float(annualized_return)
    result['annualized_vol'] = float(annualized_vol)
    result['sharpe_approx'] = float(sharpe_daily * np.sqrt(freq_per_year))

    # 4) 累积与最大回撤
    df['cum_ret'] = np.exp(df['log_ret'].cumsum())  # 累积乘积形式
    df['cum_peak'] = df['cum_ret'].cummax()
    df['drawdown'] = df['cum_ret'] / df['cum_peak'] - 1
    max_drawdown = df['drawdown'].min()
    result['max_drawdown'] = float(max_drawdown)

    # 回撤持续期（最长连续下跌天数）
    dd_mask = df['drawdown'] < 0
    longest_drawdown_duration = 0
    cur = 0
    for v in dd_mask:
        if v:
            cur += 1
            longest_drawdown_duration = max(longest_drawdown_duration, cur)
        else:
            cur = 0
    result['max_drawdown_duration_days'] = int(longest_drawdown_duration)

    # 5) VaR 与 ES（历史法 + 参数法）
    # a) 历史 VaR（以 log returns）
    var_95_hist = np.percentile(df['log_ret'], 5)
    var_99_hist = np.percentile(df['log_ret'], 1)

    # b) 参数化 VaR（正态假设）
    var_95_param = mean_daily + Z_95 * std_daily
    var_99_param = mean_daily + Z_99 * std_daily

    # ES（历史）：平均低于 VaR 的损失
    es_95_hist = df['log_ret'][df['log_ret'] <= var_95_hist].mean()
    es_99_hist = df['log_ret'][df['log_ret'] <= var_99_hist].mean()

    result.update({
        'VaR_95_hist': float(var_95_hist),
        'VaR_99_hist': float(var_99_hist),
        'VaR_95_param': float(var_95_param),
        'VaR_99_param': float(var_99_param),
        'ES_95_hist': float(es_95_hist) if not np.isnan(es_95_hist) else None,
        'ES_99_hist': float(es_99_hist) if not np.isnan(es_99_hist) else None
    })

    # 6) 趋势与动量
    # 线性回归斜率 (对 log close 做回归，以天序号作 x)
    y = np.log(df['close']).values # type: ignore
    x = np.arange(len(y))
    if len(x) >= 2:
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        # 将斜率转换成年化收益近似: exp(slope * freq) - 1
        approx_annual_slope = math.exp(slope * freq_per_year) - 1
    else:
        slope = 0.0
        approx_annual_slope = 0.0

    # 最近 20/60 天动量
    mom_20 = df['log_ret'].rolling(window=min(20, len(df))).sum().iloc[-1]
    mom_60 = df['log_ret'].rolling(window=min(60, len(df))).sum().iloc[-1]

    result.update({
        'trend_slope_daily': float(slope),
        'trend_approx_annual_return': float(approx_annual_slope),
        'momentum_20d': float(mom_20),
        'momentum_60d': float(mom_60)
    })

    # 7) 波动率制度（vol regime）
    hist_vol_ma = df['log_ret'].rolling(window=min(63, len(df))).std().mean()  # approx 3-month mean vol
    current_vol = std_daily
    vol_regime_ratio = current_vol / (hist_vol_ma + 1e-12)
    result['vol_regime_ratio'] = float(vol_regime_ratio)

    # 8) 置信区间估计（若存在多个样本路径可用样本直接估计；否则用 bootstrap）
    # 检查是否有多路径：pred_df 可能来自 sample_count>1 并合并在一张表；我们默认此函数接收单一路径。
    # 使用 bootstrap 对未来 T 终值进行不确定性估计（对 log_ret 进行重抽样）
    final_price = df['close'].iloc[-1]
    boot_final_prices = []
    rng = np.random.default_rng(42)
    for _ in range(bootstrap_iters):
        resampled = rng.choice(df['log_ret'].values, size=len(df), replace=True) # type: ignore
        final = final_price * np.exp(resampled.sum())
        boot_final_prices.append(final)
    boot_final_prices = np.array(boot_final_prices)
    ci_lower = np.percentile(boot_final_prices, 2.5)
    ci_upper = np.percentile(boot_final_prices, 97.5)
    median_final = np.median(boot_final_prices)

    result.update({
        'final_price_median_boot': float(median_final),
        'final_price_95ci_lower': float(ci_lower),
        'final_price_95ci_upper': float(ci_upper)
    })

    # 9) 仓位建议（基于规则）
    # 规则示例（可按需求调参）
    # - 强买入: trend positive, low vol_regime_ratio < 0.9, max_drawdown small > -0.05
    # - 买入: trend moderately positive, vol moderate
    # - 中性: trend neutral or vol high
    # - 卖出: trend negative or max_drawdown > 0.1 or VaR_95_param < -0.02
    score = 0.0
    # trend contribution
    if approx_annual_slope > 0.05:
        score += 1.0
    elif approx_annual_slope > 0.02:
        score += 0.5
    elif approx_annual_slope < -0.02:
        score -= 0.5
    elif approx_annual_slope < -0.05:
        score -= 1.0

    # vol penalty
    if vol_regime_ratio < 0.9:
        score += 0.5
    elif vol_regime_ratio > 1.2:
        score -= 0.7

    # drawdown penalty
    if max_drawdown < -0.15:
        score -= 1.0
    elif max_drawdown < -0.08:
        score -= 0.5

    # VaR penalty (daily)
    if result['VaR_95_param'] < -0.03:
        score -= 0.5

    # map score to advice
    if score >= 1.5:
        advice = "strong_buy"
        position_suggestion = "aggressive"  # 可考虑较大仓位
    elif score >= 0.5:
        advice = "buy"
        position_suggestion = "moderate"
    elif score > -0.5:
        advice = "hold"
        position_suggestion = "neutral"
    elif score > -1.5:
        advice = "reduce"
        position_suggestion = "defensive"
    else:
        advice = "sell"
        position_suggestion = "close"

    result.update({
        'score': float(score),
        'advice': advice,
        'position_suggestion': position_suggestion
    })

    # 10) 生成自然语言简短报告（可直接写入 AgentState）
    report_lines = []
    report_lines.append(f"预测区间长度: {n} 个交易日，最终预测价: {df['close'].iloc[-1]:.4f}")
    report_lines.append(f"近似年化收益: {annualized_return:.2%}，年化波动: {annualized_vol:.2%}")
    report_lines.append(f"最大回撤: {max_drawdown:.2%}，最长回撤持续期: {longest_drawdown_duration} 日")
    report_lines.append(f"VaR(95%, param): {result['VaR_95_param']:.2%} / VaR(95%, hist): {result['VaR_95_hist']:.2%}")
    report_lines.append(f"趋势斜率(年化近似): {approx_annual_slope:.2%}，20日动量: {mom_20:.2%}")
    report_lines.append(f"波动率制度比 (当前/历史3月均): {vol_regime_ratio:.2f}")
    report_lines.append(f"仓位建议: {advice} (档位: {position_suggestion})")
    report_lines.append(f"价格 95% CI (bootstrap): [{ci_lower:.4f}, {ci_upper:.4f}]")

    result['text_report'] = "\n".join(report_lines)

    # 11) 返回结构
    result['summary_table'] = {
        'last_price': float(df['close'].iloc[-1]),
        'median_final_price_boot': float(median_final),
        'ci_95_lower': float(ci_lower),
        'ci_95_upper': float(ci_upper),
        'advice': advice,
        'position': position_suggestion
    }

    return result


#
# # -----------------------------------------------------------------
# # 7. 执行入口 (关键修改)
# # -----------------------------------------------------------------
# if __name__ == "__main__":
#
#     # # --- 示例调用 1 (使用特定日期，覆盖默认值) ---
#     logger.info("--- [示例 1] 运行特定日期预测 (601318) ---")
#     hist_df_1, pred_df_1 = kronos_predict(
#         symbol="601519", # 大智慧
#         start_date_str="2020-01-01",
#         end_date_str="2025-05-15",
#         pred_len=100  # 覆盖默认的 90
#     )
#     print(pred_df_1)
#     result = analyze_prediction_df(pred_df_1, freq_per_year=252, bootstrap_iters=2000) # type: ignore
#     print(result)
#     logger.info("--- [示例 1] 完成 ---\n")
#
#     # --- 示例调用 2 (使用默认日期：近3年 -> 预测 90 天) ---
#     logger.info("--- [示例 2] 运行默认日期预测 (300750) ---")
#     #  使用原先被注释掉的 '300750' 作为新默认逻辑的示例
#     # 不传递 start/end/pred_len 参数，将自动使用默认值
#     # hist_df_2, pred_df_2 = kronos_predict(
#     #     symbol="601519" # 大智慧
#     #     # start_date_str=None, (使用默认值: 3年前)
#     #     # end_date_str=None, (使用默认值: 今天)
#     #     # pred_len=90 (使用默认值)
#     # )
#     logger.info("--- [示例 2] 完成 ---\n")


# -----------------------------------------------------------------
# 6. 主预测函数 (已修改)
# -----------------------------------------------------------------
from typing import Optional, Tuple
import plotly.graph_objs as go  # 确保 go 已导入


def kronos_predict(
        symbol: str,
        start_date_str: Optional[str] = None,
        end_date_str: Optional[str] = None,
        pred_len: int = 90,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[go.Figure]]:  # <--- 修改返回类型
    """
    主预测函数：获取、清理、预测并绘图。

    (已修改) 现在返回 (history_df, prediction_df, plotly_figure_object)
    """
    _lazy_load_predictor()
    if predictor is None:
        logger.error("预测器未初始化，无法执行预测。")
        return None, None, None  # <--- [修改] 返回 3 个值

    # --- 默认日期逻辑处理 ---
    if end_date_str is None:
        end_ts = datetime.now()
        end_date_str_log = end_ts.strftime("%Y-%m-%d")
        logger.info(f"未指定 end_date_str，使用默认值 (今天): {end_date_str_log}")
    else:
        end_ts = pd.Timestamp(end_date_str)
        end_date_str_log = end_date_str

    if start_date_str is None:
        start_ts = end_ts - timedelta(days=3 * 365)
        start_date_str_log = start_ts.strftime("%Y-%m-%d")
        logger.info(f"未指定 start_date_str，使用默认值 (3年前): {start_date_str_log}")
    else:
        start_ts = pd.Timestamp(start_date_str)
        start_date_str_log = start_date_str

        # --- (多余的检查，但保持一致) ---
    if predictor is None:
        logger.error("预测器未初始化，无法执行预测。")
        return None, None, None  # <--- [修改] 返回 3 个值

    logger.info(
        f"--- [START] 开始为 {symbol} 执行预测 (周期: {start_date_str_log} to {end_date_str_log}, 预测 {pred_len} 天) ---")

    # 1. 数据获取
    df_raw = get_and_process_data(symbol, start_ts, end_ts)

    if df_raw.empty:
        logger.error(f"获取数据失败 {symbol}。")
        return None, None, None  # <--- [修改] 返回 3 个值

    # 2. 数据清理
    df_cleaned = clear_data(df_raw.copy())

    if df_cleaned.empty or len(df_cleaned) < 10:
        logger.error(f"清理后数据不足10条 {symbol} (剩余 {len(df_cleaned)} 条)，无法预测。")
        # [修改] 即使失败也返回清理后的df，但 pred 和 fig 为 None
        return df_cleaned, None, None

        # 3. 在清理数据 *之后* 创建时间戳
    df_cleaned["timestamps"] = df_cleaned["date"]
    df_cleaned["timestamps"] = pd.to_datetime(df_cleaned["timestamps"])
    logger.info(f"已使用 'date' 列中的实际交易日作为 'timestamps'。")

    # 4. 准备预测输入
    last_timestamp = df_cleaned["timestamps"].iloc[-1]
    logger.info(f"历史数据中的最后一个 'timestamp' 是: {last_timestamp.date()}")

    x_timestamp = pd.Series(df_cleaned["timestamps"])
    y_timestamp = pd.Series(
        pd.date_range(
            start=last_timestamp + pd.Timedelta(days=1),
            periods=pred_len,
            freq="B"
        )
    )

    logger.info(f"预测将从 {y_timestamp.iloc[0].date()} 开始 (使用 'B' 频率跳过周末)。")

    model_input_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamps']
    df_for_model = df_cleaned[model_input_cols]

    logger.info(f"准备调用预测器... 历史数据 {len(df_for_model)} 条 (Volume 已 Log 变换)。")

    # 5. 调用 predict
    fig: Optional[go.Figure] = None  # <--- 初始化 fig
    try:
        with torch.inference_mode():
            pred_df = predictor.predict(
                df=df_for_model,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=T,
                top_p=top_p,
                sample_count=sample_count
            )

        if 'volume' in pred_df.columns:
            pred_df['volume'] = np.expm1(pred_df['volume'])
            logger.info("  - (关键) 已反转预测结果 'volume' 的 log1p 变换 (expm1)")

        logger.info("预测成功。")
        print("\n--- 预测结果 (Head) ---")
        print(pred_df[["open", "high", "low", "close", "volume"]].head())
        print("------------------------\n")

        # 6. 绘图
        logger.info("生成预测对比图...")
        fig = build_prediction_figure(df_cleaned, pred_df)  # <--- fig 在这里被赋值

        #  保存图表到本地文件 ---
        try:
            global output_dir
            safe_end_date_str = end_ts.strftime("%Y%m%d")
            filename = f"{symbol}_{safe_end_date_str}_pred_{pred_len}d.html"
            save_path = os.path.join(output_dir, filename)

            fig.write_html(save_path)

            logger.info(f"预测图表已成功保存到: {save_path}")
            print(f"f 预测图表已成功保存到: {save_path}")

        except Exception as e:
            logger.error(f"保存 HTML 图表失败: {e}")
            print(f"❌ 保存 HTML 图表失败: {e}")

        logger.info(f"--- [END] 预测完成 {symbol} ---")

        # <--- [关键修改] ---
        return df_cleaned, pred_df, fig  # <--- 返回 fig 对象

    except Exception as e:
        logger.exception(f"预测或绘图过程中发生严重错误: {e}")
        # <--- [关键修改] ---
        # 返回 df_cleaned（如果已生成），但 pred_df 和 fig 为 None
        return df_cleaned, None, fig

    # -----------------------------------------------------------------


# 7. 执行入口 (已修改)
# -----------------------------------------------------------------
if __name__ == "__main__":

    # --- 示例调用 1 (使用特定日期，覆盖默认值) ---
    logger.info("--- [示例 1] 运行特定日期预测 (601519) ---")

    # [修改] 接收 fig_1
    hist_df_1, pred_df_1, fig_1 = kronos_predict(
        symbol="601519",  # 大智慧
        start_date_str="2020-01-01",
        end_date_str="2025-05-15",
        pred_len=100  # 覆盖默认的 90
    )

    # 检查预测是否成功
    if pred_df_1 is not None:
        logger.info("示例 1 预测成功。")
        result = analyze_prediction_df(pred_df_1, freq_per_year=252, bootstrap_iters=2000)
        print("\n--- 预测分析报告 (示例 1) ---")
        print(result['text_report'])
        print("---------------------------------\n")

        # 如果您在 Jupyter Notebook 中运行，可以取消注释下面这行来显示图表
        fig_1.show()
    else:
        logger.error("示例 1 预测失败。")

    logger.info("--- [示例 1] 完成 ---\n")

    # --- 示例调用 2 (使用默认日期：近3年 -> 预测 90 天) ---
    logger.info("--- [示例 2] 运行默认日期预测 (601519) ---")

    # [修改] 接收 fig_2
    hist_df_2, pred_df_2, fig_2 = kronos_predict(
        symbol="601519"  # 大智慧
        # start_date_str=None, (使用默认值: 3年前)
        # end_date_str=None, (使用默认值: 今天)
        # pred_len=90 (使用默认值)
    )

    if pred_df_2 is not None:
        logger.info("示例 2 预测成功。")
        result_2 = analyze_prediction_df(pred_df_2, freq_per_year=252, bootstrap_iters=2000)
        print("\n--- 预测分析报告 (示例 2) ---")
        print(result_2['text_report'])
        print("---------------------------------\n")
        fig_2.show()
    else:
        logger.error("示例 2 预测失败。")

    logger.info("--- [示例 2] 完成 ---\n")