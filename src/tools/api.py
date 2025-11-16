from typing import Dict, Any, List
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import json
import numpy as np
# from ..utils.logging_config import setup_logger

import os
import time
import logging
from typing import Optional


def setup_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """设置统一的日志配置

    Args:
        name: logger的名称
        log_dir: 日志文件目录，如果为None则使用默认的logs目录

    Returns:
        配置好的logger实例
    """
    # 设置 root logger 的级别为 DEBUG
    logging.getLogger().setLevel(logging.DEBUG)

    # 获取或创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # logger本身记录DEBUG级别及以上
    logger.propagate = False  # 防止日志消息传播到父级logger

    # 如果已经有处理器，不再添加
    if logger.handlers:
        return logger

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 控制台只显示INFO及以上级别

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # 创建文件处理器
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # 文件记录DEBUG级别及以上的日志
    file_handler.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# 预定义的图标
SUCCESS_ICON = "✓"
ERROR_ICON = "✗"
WAIT_ICON = "🔄"








# 设置日志记录
logger = setup_logger('api_tool')

def get_stock_prefix(symbol: str) -> str:
    """根据股票代码判断上海(sh)或深圳(sz)交易所"""
    if symbol.startswith('6') or symbol.startswith('9') or symbol.startswith('11'):
        # 60 (主板), 68 (科创板), 900 (B股)
        return f"sh{symbol}"
    else:
        # 00 (主板), 30 (创业板), 20 (中小板), 08 (配股)
        return f"sz{symbol}"

def get_financial_metrics(symbol: str) -> List[Dict[str, Any]]:
    """获取财务指标数据（增强稳定性与防护性）"""
    logger.info(f"获取股票代码为 {symbol} 的财务指标数据...")
    try:
        # 辅助函数：安全取 float（处理 None、'--'、百分号字符串等）
        def safe_float(x, default=0.0):
            try:
                if x is None:
                    return float(default)
                if isinstance(x, str):
                    s = x.strip()
                    if s in ("", "--", "-", "—", "NaN", "nan"):
                        return float(default)
                    # 处理带% 的百分比（例如 "12.34%" 或 "12.34 %")
                    if s.endswith("%"):
                        s2 = s.replace("%", "").strip()
                        return float(s2) / 100.0
                    return float(s)
                if pd.isna(x):
                    return float(default)
                return float(x)
            except Exception:
                return float(default)

        # 1) 尝试获取估值 / 市值信息（ak.stock_value_em 可能返回 DataFrame）
        cap_df = None
        try:
            tmp = ak.stock_value_em(symbol)
            if tmp is not None and hasattr(tmp, "empty"):
                if not tmp.empty:
                    # 如果返回的是 DataFrame，使用最后一行或尝试转置为一行
                    cap_df = tmp
                else:
                    cap_df = None
            else:
                cap_df = None
        except Exception as e:
            logger.debug(f"调用 ak.stock_value_em 失败: {e}")
            cap_df = None

        # 备用：尝试使用 stock_individual_info_em 获取 key-value 表格
        cap_kv_df = None
        try:
            tmp2 = ak.stock_individual_info_em(symbol)
            if tmp2 is not None and hasattr(tmp2, "empty") and not tmp2.empty:
                # 通常返回两列 (item, value)
                if "item" in tmp2.columns and "value" in tmp2.columns:
                    cap_kv_df = tmp2.set_index("item")["value"].to_frame().T
                else:
                    cap_kv_df = tmp2
            else:
                cap_kv_df = None
        except Exception as e:
            logger.debug(f"调用 ak.stock_individual_info_em 失败: {e}")
            cap_kv_df = None

        # 选择 cap_data（优先 cap_df，如果不行再用 cap_kv_df）
        cap_data = None
        if cap_df is not None:
            cap_data = cap_df
        elif cap_kv_df is not None:
            cap_data = cap_kv_df

        if cap_data is None or cap_data.empty:
            logger.warning(f"× {symbol} 没有可用的股票市值/估值数据 (cap_data empty)")
            # 继续，但后续会使用默认值
        else:
            logger.info(f"✓ {symbol} 的市值/估值数据获取成功 (columns: {list(cap_data.columns)})")

        # 2) 获取新浪财务指标（ak.stock_financial_analysis_indicator）
        current_year = datetime.now().year
        financial_data = None
        try:
            tmp_fin = ak.stock_financial_analysis_indicator(symbol=symbol, start_year=str(current_year-1))
            if tmp_fin is not None and hasattr(tmp_fin, "empty") and not tmp_fin.empty:
                tmp_fin["日期"] = pd.to_datetime(tmp_fin["日期"], errors="coerce")
                tmp_fin = tmp_fin.sort_values("日期", ascending=False)
                financial_data = tmp_fin
            else:
                financial_data = None
        except Exception as e:
            logger.debug(f"调用 ak.stock_financial_analysis_indicator 失败: {e}")
            financial_data = None

        if financial_data is None:
            logger.warning(f"× {symbol} 没有可用的新浪财务指标数据")
            latest_financial = pd.Series()
        else:
            latest_financial = financial_data.iloc[0] if len(financial_data) > 0 else pd.Series()
            logger.info(f"✓ {symbol} 的财务指标获取成功，共{len(financial_data) if financial_data is not None else 0}条记录 (最后一条日期为{latest_financial.get('日期', None)})")

        # 3) 获取利润表（用于 revenue）
        latest_income = pd.Series()
        try:
            stock_code = get_stock_prefix(symbol)
            income_statement = ak.stock_financial_report_sina(stock=stock_code, symbol="利润表")
            if income_statement is not None and hasattr(income_statement, "empty") and not income_statement.empty:
                latest_income = income_statement.iloc[0]
                logger.info(f"✓ {symbol} 的利润表数据获取成功")
            else:
                logger.warning(f"× {symbol} 的利润表数据为空或缺失")
        except Exception as e:
            logger.debug(f"调用 ak.stock_financial_report_sina(利润表) 失败: {e}")

        # 4) 整合并构建指标（全部使用 safe_float）
        try:
            # 从 cap_data 中安全读取字段（列名可能有差异）
            def cap_get(field_names):
                """尝试多种列名并返回第一个存在的值"""
                if cap_data is None or cap_data.empty:
                    return 0.0
                for fn in field_names:
                    if fn in cap_data.columns:
                        try:
                            val = cap_data[fn].iloc[0]
                            return val
                        except Exception:
                            continue
                return 0.0

            total_market_val = safe_float(cap_get(["总市值", "总市值(元)", "总市值(万)", "市值"]))
            float_market_val = safe_float(cap_get(["流通市值", "流通市值(元)", "流通市值(万)"]))
            general_capital = safe_float(cap_get(["总股本", "总股本(股)"]))
            float_capital = safe_float(cap_get(["流通股本", "流通股本(股)"]))
            pe_ratio = safe_float(cap_get(["PE(TTM)", "市盈率"]))
            price_to_book = safe_float(cap_get(["市净率", "PB"]))

            # revenue 从利润表中取 营业总收入 / 营业收入 等字段候选
            revenue_candidates = ["营业总收入", "营业收入", "主营业务收入"]
            revenue = 0.0
            for rc in revenue_candidates:
                if rc in latest_income.index:
                    revenue = safe_float(latest_income.get(rc, 0.0))
                    if revenue != 0.0:
                        break

            # 若 revenue 为 0 则 price_to_sales 设为 0 避免除0
            price_to_sales = (total_market_val / revenue) if revenue not in (0, 0.0) else 0.0

            # 百分数指标从 latest_financial 中读取（字段名可能有差异）
            def fin_get_pct(candidates):
                for c in candidates:
                    if c in latest_financial.index:
                        return safe_float(latest_financial.get(c, 0.0))  # safe_float 会处理 '%' 并返回小数
                return 0.0

            return_on_equity = fin_get_pct(["净资产收益率(%)", "净资产收益率"])
            net_margin = fin_get_pct(["销售净利率(%)", "销售净利率", "净利率(%)", "净利率"])
            operating_margin = fin_get_pct(["营业利润率(%)", "营业利润率"])

            revenue_growth = fin_get_pct(["主营业务收入增长率(%)", "主营业务收入增长率"])
            earnings_growth = fin_get_pct(["净利润增长率(%)", "净利润增长率"])
            book_value_growth = fin_get_pct(["净资产增长率(%)", "净资产增长率"])

            current_ratio = safe_float(latest_financial.get("流动比率", 0.0))
            debt_to_equity = fin_get_pct(["资产负债率(%)", "资产负债率"])

            free_cash_flow_per_share = safe_float(latest_financial.get("每股经营性现金流(元)", 0.0))
            earnings_per_share = safe_float(latest_financial.get("加权每股收益(元)", 0.0))

            all_metrics = {
                "market_cap": total_market_val,
                "float_market_cap": float_market_val,
                "general_capital": general_capital,
                "float_capital": float_capital,
                "revenue": revenue,
                "net_income": safe_float(latest_income.get("净利润", 0.0)),
                "return_on_equity": return_on_equity,
                "net_margin": net_margin,
                "operating_margin": operating_margin,
                "revenue_growth": revenue_growth,
                "earnings_growth": earnings_growth,
                "book_value_growth": book_value_growth,
                "current_ratio": current_ratio,
                "debt_to_equity": debt_to_equity,
                "free_cash_flow_per_share": free_cash_flow_per_share,
                "earnings_per_share": earnings_per_share,
                "pe_ratio": pe_ratio,
                "price_to_book": price_to_book,
                "price_to_sales": price_to_sales,
            }

            agent_metrics = {
                "market_cap": all_metrics["market_cap"],
                "float_market_cap": all_metrics["float_market_cap"],
                "general_capital": all_metrics["general_capital"],
                "float_capital": all_metrics["float_capital"],
                "return_on_equity": all_metrics["return_on_equity"],
                "net_margin": all_metrics["net_margin"],
                "operating_margin": all_metrics["operating_margin"],
                "revenue_growth": all_metrics["revenue_growth"],
                "earnings_growth": all_metrics["earnings_growth"],
                "book_value_growth": all_metrics["book_value_growth"],
                "current_ratio": all_metrics["current_ratio"],
                "debt_to_equity": all_metrics["debt_to_equity"],
                "free_cash_flow_per_share": all_metrics["free_cash_flow_per_share"],
                "earnings_per_share": all_metrics["earnings_per_share"],
                "pe_ratio": all_metrics["pe_ratio"],
                "price_to_book": all_metrics["price_to_book"],
                "price_to_sales": all_metrics["price_to_sales"],
            }

            logger.info(f"{symbol} 的财务指标数据获取并整合完成")
            return [agent_metrics]

        except Exception as e:
            logger.error(f"× {symbol} 的财务指标数据整合失败：{e}")
            logger.debug(traceback.format_exc())
            return [{}]

    except Exception as e:
        logger.error(f"Error getting financial indicators: {e}")
        logger.debug(traceback.format_exc())
        return [{}]



def get_financial_statements(symbol: str) -> Dict[str, Any]:
    """获取财务报表数据"""
    logger.info(f"获取股票代码为 {symbol} 的财务报表数据...")
    try:
        # 获取资产负债表数据
        # logger.info("Fetching balance sheet...")
        try:
            stock_code = get_stock_prefix(symbol)
            balance_sheet = ak.stock_financial_report_sina(
                stock=stock_code, symbol="资产负债表")
            if not balance_sheet.empty:
                latest_balance = balance_sheet.iloc[0]
                previous_balance = balance_sheet.iloc[1] if len(
                    balance_sheet) > 1 else balance_sheet.iloc[0]
                logger.info(f"✓ {symbol} 的资产负债表数据获取成功")
            else:
                logger.warning(f"× {symbol} 的资产负债表数据获取失败")
                logger.error(f"× {symbol} 没有找到资产负债表数据")
                latest_balance = pd.Series()
                previous_balance = pd.Series()
        except Exception as e:
            logger.warning(f"× {symbol} 的资产负债表数据获取失败")
            logger.error(f"失败原因: {e}")
            latest_balance = pd.Series()
            previous_balance = pd.Series()

        # 获取利润表数据
        # logger.info("Fetching income statement...")
        try:
            stock_code = get_stock_prefix(symbol)
            income_statement = ak.stock_financial_report_sina(
                stock=stock_code, symbol="利润表")
            if not income_statement.empty:
                latest_income = income_statement.iloc[0]
                previous_income = income_statement.iloc[1] if len(
                    income_statement) > 1 else income_statement.iloc[0]
                logger.info(f"✓ {symbol} 的利润表数据获取成功")
            else:
                logger.warning(f"× {symbol} 的利润表数据获取失败")
                logger.error(f"× {symbol} 没有找到利润表数据")
                latest_income = pd.Series()
                previous_income = pd.Series()
        except Exception as e:
            logger.warning(f"× {symbol} 的利润表数据获取失败")
            logger.error(f"失败原因: {e}")
            latest_income = pd.Series()
            previous_income = pd.Series()

        # 获取现金流量表数据
        logger.info("获取现金流量表...")
        try:
            stock_code = get_stock_prefix(symbol)
            cash_flow = ak.stock_financial_report_sina(
                stock=stock_code, symbol="现金流量表")
            if not cash_flow.empty:
                latest_cash_flow = cash_flow.iloc[0]
                previous_cash_flow = cash_flow.iloc[1] if len(
                    cash_flow) > 1 else cash_flow.iloc[0]
                logger.info(f"✓ {symbol} 的现金流量表数据获取成功")
            else:
                logger.warning(f"× {symbol} 的现金流量表数据获取失败")
                logger.error(f"× {symbol} 没有找到现金流量表数据")
                latest_cash_flow = pd.Series()
                previous_cash_flow = pd.Series()
        except Exception as e:
            logger.warning(f"× {symbol} 的现金流量表数据获取失败")
            logger.error(f"失败原因: {e}")
            latest_cash_flow = pd.Series()
            previous_cash_flow = pd.Series()

        # 构建财务数据
        line_items = []
        try:
            # 处理最新期间数据
            current_item = {
                # 从利润表获取
                "net_income": float(latest_income.get("净利润", 0)),
                "operating_revenue": float(latest_income.get("营业总收入", 0)),
                "operating_profit": float(latest_income.get("营业利润", 0)),

                # 从资产负债表计算营运资金
                "working_capital": float(latest_balance.get("流动资产合计", 0)) - float(latest_balance.get("流动负债合计", 0)),

                # 从现金流量表获取
                "depreciation_and_amortization": float(latest_cash_flow.get("固定资产折旧、油气资产折耗、生产性生物资产折旧", 0)),
                "capital_expenditure": abs(float(latest_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0))),
                "free_cash_flow": float(latest_cash_flow.get("经营活动产生的现金流量净额", 0)) - abs(float(latest_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0)))
            }
            line_items.append(current_item)
            logger.info(f"✓ {symbol} 本期的财务报表数据获取并整合完成")

            # 处理上一期间数据
            previous_item = {
                "net_income": float(previous_income.get("净利润", 0)),
                "operating_revenue": float(previous_income.get("营业总收入", 0)),
                "operating_profit": float(previous_income.get("营业利润", 0)),
                "working_capital": float(previous_balance.get("流动资产合计", 0)) - float(previous_balance.get("流动负债合计", 0)),
                "depreciation_and_amortization": float(previous_cash_flow.get("固定资产折旧、油气资产折耗、生产性生物资产折旧", 0)),
                "capital_expenditure": abs(float(previous_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0))),
                "free_cash_flow": float(previous_cash_flow.get("经营活动产生的现金流量净额", 0)) - abs(float(previous_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0)))
            }
            line_items.append(previous_item)
            logger.info(f"✓ {symbol} 上一期的财务报表数据获取并整合完成")

        except Exception as e:
            logger.error(f"× {symbol} 的财务报表数据获取失败: {e}")
            default_item = {
                "net_income": 0,
                "operating_revenue": 0,
                "operating_profit": 0,
                "working_capital": 0,
                "depreciation_and_amortization": 0,
                "capital_expenditure": 0,
                "free_cash_flow": 0
            }
            line_items = [default_item, default_item]

        return line_items

    except Exception as e:
        logger.error(f"Error getting financial statements: {e}")
        default_item = {
            "net_income": 0,
            "operating_revenue": 0,
            "operating_profit": 0,
            "working_capital": 0,
            "depreciation_and_amortization": 0,
            "capital_expenditure": 0,
            "free_cash_flow": 0
        }
        return [default_item, default_item]


# def get_market_data(symbol: str) -> Dict[str, Any]:
#     """获取市场数据"""
#     logger.info(f"获取股票代码为 {symbol} 的市值数据...")
#     try:
#         # 获取实时行情
#         # realtime_data = ak.stock_zh_a_spot()
#         logger.info(f"开始获取 {symbol} 的实时行情数据...")
#         realtime_data_df = ak.stock_bid_ask_em(symbol=symbol)
#         if realtime_data_df is None or realtime_data_df.empty:
#             logger.warning(f"× {symbol} 没有可用的 stock_bid_ask_em 数据")
#             return {}
#         stock_data = realtime_data_df.set_index('item')['value']
#         logger.info(f"✓ {symbol} 的市场数据获取成功")
#
#         # 获取市值数据
#         cap_data = ak.stock_individual_info_em(symbol)
#         cap_data = cap_data.set_index('item')['value'].to_frame().T
#         if cap_data is None or cap_data.empty:
#             logger.warning(f"× {symbol} 没有可用的股票市值数据")
#             return [{}]
#         logger.info(f"✓ {symbol} 的市值数据获取成功")
#
#         return {
#             "market_cap": float(cap_data["总市值"].iloc[0]),
#             # "volume": float(stock_data.get("成交量", 0)), ## ak.stock_bid_ask_em 这个单次请求接口返回的没有成交量字段，用 "量比"代替
#             "volume_ratio": float(stock_data.get("量比", 0)),
#             "general_capital": float(cap_data.get("总股本", 0))
#             # "fifty_two_week_high": float(stock_data.get("52周最高", 0)),
#             # "fifty_two_week_low": float(stock_data.get("52周最低", 0))
#         }
#
#     except Exception as e:
#         logger.error(f"Error getting market data: {e}")
#         return {}


def get_price_history(symbol: str, start_date: str = None, end_date: str = None, adjust: str = "qfq") -> pd.DataFrame:
    """获取历史价格数据

    Args:
        symbol: 股票代码
        start_date: 开始日期，格式：YYYY-MM-DD，如果为None则默认获取过去一年的数据
        end_date: 结束日期，格式：YYYY-MM-DD，如果为None则使用昨天作为结束日期
        adjust: 复权类型，可选值：
               - "": 不复权
               - "qfq": 前复权（默认）
               - "hfq": 后复权

    Returns:
        包含以下列的DataFrame：
        - date: 日期
        - open: 开盘价
        - high: 最高价
        - low: 最低价
        - close: 收盘价
        - volume: 成交量（手）
        - amount: 成交额（元）
        - amplitude: 振幅（%）
        - pct_change: 涨跌幅（%）
        - change_amount: 涨跌额（元）
        - turnover: 换手率（%）

        技术指标：
        - momentum_1m: 1个月动量
        - momentum_3m: 3个月动量
        - momentum_6m: 6个月动量
        - volume_momentum: 成交量动量
        - historical_volatility: 历史波动率
        - volatility_regime: 波动率区间
        - volatility_z_score: 波动率Z分数
        - atr_ratio: 真实波动幅度比率
        - hurst_exponent: 赫斯特指数
        - skewness: 偏度
        - kurtosis: 峰度
    """
    try:
        # 获取当前日期和昨天的日期
        current_date = datetime.now()
        yesterday = current_date - timedelta(days=1)

        # 如果没有提供日期，默认使用昨天作为结束日期
        if not end_date:
            end_date = yesterday  # 使用昨天作为结束日期
        else:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            # 确保end_date不会超过昨天
            if end_date > yesterday:
                end_date = yesterday

        if not start_date:
            start_date = end_date - timedelta(days=365)  # 默认获取一年的数据
        else:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        logger.info(f"获取股票代码为 {symbol} 的价格数据...")
                    # f"，时间范围{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"

        def get_and_process_data(start_date, end_date):
            """获取并处理数据，包括重命名列等操作"""
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust=adjust
            )

            if df is None or df.empty:
                return pd.DataFrame()

            # 重命名列以匹配技术分析代理的需求
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_change",
                "涨跌额": "change_amount",
                "换手率": "turnover"
            })

            # 确保日期列为datetime类型
            df["date"] = pd.to_datetime(df["date"])
            return df

        # 获取历史行情数据
        df = get_and_process_data(start_date, end_date)

        if df is None or df.empty:
            logger.warning(
                f"Warning: No price history data found for {symbol}")
            return pd.DataFrame()

        # 检查数据量是否足够
        min_required_days = 120  # 至少需要120个交易日的数据
        if len(df) < min_required_days:
            logger.warning(
                f"Warning: Insufficient data ({len(df)} days) for all technical indicators")
            logger.info("Attempting to fetch more data...")

            # 扩大时间范围到2年
            start_date = end_date - timedelta(days=730)
            df = get_and_process_data(start_date, end_date)

            if len(df) < min_required_days:
                logger.warning(
                    f"Warning: Even with extended time range, insufficient data ({len(df)} days)")

        # 计算动量指标
        df["momentum_1m"] = df["close"].pct_change(periods=20)  # 20个交易日约等于1个月
        df["momentum_3m"] = df["close"].pct_change(periods=60)  # 60个交易日约等于3个月
        df["momentum_6m"] = df["close"].pct_change(
            periods=120)  # 120个交易日约等于6个月

        # 计算成交量动量（相对于20日平均成交量的变化）
        df["volume_ma20"] = df["volume"].rolling(window=20).mean()
        df["volume_momentum"] = df["volume"] / df["volume_ma20"]

        # 计算波动率指标
        # 1. 历史波动率 (20日)
        returns = df["close"].pct_change()
        df["historical_volatility"] = returns.rolling(
            window=20).std() * np.sqrt(252)  # 年化

        # 2. 波动率区间 (相对于过去120天的波动率的位置)
        volatility_120d = returns.rolling(window=120).std() * np.sqrt(252)
        vol_min = volatility_120d.rolling(window=120).min()
        vol_max = volatility_120d.rolling(window=120).max()
        vol_range = vol_max - vol_min
        df["volatility_regime"] = np.where(
            vol_range > 0,
            (df["historical_volatility"] - vol_min) / vol_range,
            0  # 当范围为0时返回0
        )

        # 3. 波动率Z分数
        vol_mean = df["historical_volatility"].rolling(window=120).mean()
        vol_std = df["historical_volatility"].rolling(window=120).std()
        df["volatility_z_score"] = (
            df["historical_volatility"] - vol_mean) / vol_std

        # 4. ATR比率
        tr = pd.DataFrame()
        tr["h-l"] = df["high"] - df["low"]
        tr["h-pc"] = abs(df["high"] - df["close"].shift(1))
        tr["l-pc"] = abs(df["low"] - df["close"].shift(1))
        tr["tr"] = tr[["h-l", "h-pc", "l-pc"]].max(axis=1)
        df["atr"] = tr["tr"].rolling(window=14).mean()
        df["atr_ratio"] = df["atr"] / df["close"]

        # 计算统计套利指标
        # 1. 赫斯特指数 (使用过去120天的数据)
        def calculate_hurst(series):
            """
            计算Hurst指数。

            Args:
                series: 价格序列

            Returns:
                float: Hurst指数，或在计算失败时返回np.nan
            """
            try:
                series = series.dropna()
                if len(series) < 30:  # 降低最小数据点要求
                    return np.nan

                # 使用对数收益率
                log_returns = np.log(series / series.shift(1)).dropna()
                if len(log_returns) < 30:  # 降低最小数据点要求
                    return np.nan

                # 使用更小的lag范围
                # 减少lag范围到2-10天
                lags = range(2, min(11, len(log_returns) // 4))

                # 计算每个lag的标准差
                tau = []
                for lag in lags:
                    # 计算滚动标准差
                    std = log_returns.rolling(window=lag).std().dropna()
                    if len(std) > 0:
                        tau.append(np.mean(std))

                # 基本的数值检查
                if len(tau) < 3:  # 进一步降低最小要求
                    return np.nan

                # 使用对数回归
                lags_log = np.log(list(lags))
                tau_log = np.log(tau)

                # 计算回归系数
                reg = np.polyfit(lags_log, tau_log, 1)
                hurst = reg[0] / 2.0

                # 只保留基本的数值检查
                if np.isnan(hurst) or np.isinf(hurst):
                    return np.nan

                return hurst

            except Exception as e:
                return np.nan

        # 使用对数收益率计算Hurst指数
        log_returns = np.log(df["close"] / df["close"].shift(1))
        df["hurst_exponent"] = log_returns.rolling(
            window=120,
            min_periods=60  # 要求至少60个数据点
        ).apply(calculate_hurst)

        # 2. 偏度 (20日)
        df["skewness"] = returns.rolling(window=20).skew()

        # 3. 峰度 (20日)
        df["kurtosis"] = returns.rolling(window=20).kurt()

        # 按日期升序排序
        df = df.sort_values("date")

        # 重置索引
        df = df.reset_index(drop=True)

        logger.info(f"✓ {symbol} 的股票价格数据获取成功")

        # 检查并报告NaN值
        # nan_columns = df.isna().sum()
        # if nan_columns.any():
        #     logger.warning(
        #         "\nWarning: The following indicators contain NaN values:")
        #     for col, nan_count in nan_columns[nan_columns > 0].items():
        #         logger.warning(f"- {col}: {nan_count} records")

        return df

    except Exception as e:
        logger.error(f"Error getting price history: {e}")
        return pd.DataFrame()


def prices_to_df(prices):
    """Convert price data to DataFrame with standardized column names"""
    try:
        df = pd.DataFrame(prices)

        # 标准化列名映射
        column_mapping = {
            '收盘': 'close',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'change_percent',
            '涨跌额': 'change_amount',
            '换手率': 'turnover_rate'
        }

        # 重命名列
        for cn, en in column_mapping.items():
            if cn in df.columns:
                df[en] = df[cn]

        # 确保必要的列存在
        required_columns = ['close', 'open', 'high', 'low', 'volume']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0  # 使用0填充缺失的必要列

        return df
    except Exception as e:
        logger.error(f"Error converting price data: {str(e)}")
        # 返回一个包含必要列的空DataFrame
        return pd.DataFrame(columns=['close', 'open', 'high', 'low', 'volume'])


def get_price_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """获取股票价格数据

    Args:
        ticker: 股票代码
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD

    Returns:
        包含价格数据的DataFrame
    """
    return get_price_history(ticker, start_date, end_date)



if __name__ == "__main__":
    """
    测试入口：验证模块核心函数的基本功能。
    使用示例：
        python -m src.tools.api_tool  或 直接运行此文件
    支持命令行参数：--symbol, --start_date, --end_date
    """
    import argparse
    import traceback
    import time

    parser = argparse.ArgumentParser(description="Test api_tool core functions")
    parser.add_argument("--symbol", type=str, default="600000", help="股票代码（不带交易所前缀），例如 600000")
    parser.add_argument("--start_date", type=str, default="", help="开始日期 YYYY-MM-DD（可选）")
    parser.add_argument("--end_date", type=str, default="", help="结束日期 YYYY-MM-DD（可选，不能晚于昨天）")
    args = parser.parse_args()

    symbol = args.symbol
    start_date = args.start_date or None
    end_date = args.end_date or None

    print("\n=== api_tool 单元测试开始 ===")
    print(f"测试时间：{datetime.now().isoformat()}")
    print(f"测试股票：{symbol}, start_date={start_date}, end_date={end_date}\n")

    # 1) 测试 get_stock_prefix
    try:
        print("1) 测试 get_stock_prefix ...")
        samples = ["600000", "000001", "300750", "900901", "110123"]
        for s in samples:
            pref = get_stock_prefix(s)
            print(f"   {s} -> {pref}")
            assert pref.endswith(s), "返回的前缀字符串应以原始代码结尾"
        print("   ✅ get_stock_prefix 基本测试通过\n")
    except Exception as e:
        print("   ❌ get_stock_prefix 测试失败：", str(e))
        traceback.print_exc()

    # 2) 测试 prices_to_df（无网络依赖）
    try:
        print("2) 测试 prices_to_df (本地转换函数) ...")
        sample_prices = [
            {"收盘": 10, "开盘": 9.8, "最高": 10.2, "最低": 9.7, "成交量": 12000, "换手率": 0.5},
            {"收盘": 10.1, "开盘": 10, "最高": 10.3, "最低": 9.9, "成交量": 15000, "换手率": 0.6},
        ]
        df_test = prices_to_df(sample_prices)
        print("   转换结果 columns:", df_test.columns.tolist())
        assert all(c in df_test.columns for c in ["close", "open", "high", "low", "volume"]), "必要列缺失"
        print("   head:\n", df_test.head().to_string(index=False))
        print("   ✅ prices_to_df 基本测试通过\n")
    except Exception as e:
        print("   ❌ prices_to_df 测试失败：", str(e))
        traceback.print_exc()

    # 3) 测试 get_price_history（网络依赖 —— akshare）
    try:
        print("3) 测试 get_price_history (网络请求，可能较慢) ...")
        t0 = time.time()
        df_prices = get_price_history(symbol, start_date, end_date)
        t1 = time.time()
        print(f"   请求耗时: {t1 - t0:.2f}s")
        if df_prices is None or df_prices.empty:
            print("   ⚠️ 返回的数据为空 DataFrame（可能为网络/数据源问题或参数导致无数据）")
        else:
            print(f"   返回行数: {len(df_prices)}")
            print("   列名示例:", df_prices.columns.tolist())
            # 基本校验
            assert "date" in df_prices.columns, "缺少 date 列"
            assert "close" in df_prices.columns, "缺少 close 列"
            # 检查日期最大值不超过昨天
            max_date = df_prices["date"].max()
            yesterday = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            assert pd.to_datetime(max_date) <= pd.to_datetime(yesterday) + pd.Timedelta(days=1), "返回数据含今日或未来日期"
            print("   head:\n", df_prices.head(3).to_string(index=False))
            print("   tail:\n", df_prices.tail(3).to_string(index=False))
            # 检查技术指标列是否已计算（若数据足够）
            indicator_cols = ["momentum_1m", "historical_volatility", "atr_ratio", "hurst_exponent"]
            present_indicators = [c for c in indicator_cols if c in df_prices.columns]
            print(f"   计算得到的指标（存在）: {present_indicators}")
            print("   ✅ get_price_history 基本测试通过\n")
    except AssertionError as ae:
        print("   ❌ get_price_history 断言失败：", str(ae))
        traceback.print_exc()
    except Exception as e:
        print("   ❌ get_price_history 调用失败：", str(e))
        traceback.print_exc()

    # 4) 测试 get_price_data (wrapper)
    try:
        print("4) 测试 get_price_data (wrapper) ...")
        df_wrapper = get_price_data(symbol, start_date or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                                   end_date or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))
        if df_wrapper is None or df_wrapper.empty:
            print("   ⚠️ get_price_data 返回空（可能为网络/数据源问题）")
        else:
            print(f"   get_price_data 返回 {len(df_wrapper)} 行, 列示例: {df_wrapper.columns.tolist()[:10]}")
            print("   ✅ get_price_data 基本测试通过\n")
    except Exception as e:
        print("   ❌ get_price_data 测试失败：", str(e))
        traceback.print_exc()

    # 5) 测试 get_financial_metrics（网络依赖 —— akshare）
    try:
        print("5) 测试 get_financial_metrics (网络请求) ...")
        t0 = time.time()
        metrics_list = get_financial_metrics(symbol)
        t1 = time.time()
        print(f"   请求耗时: {t1 - t0:.2f}s")
        if not metrics_list:
            print("   ⚠️ 返回为空或列表中无有效元素")
        else:
            metrics = metrics_list[0] if isinstance(metrics_list, list) else metrics_list
            print("   返回字段示例:", list(metrics.keys())[:20])
            # 基本类型检查（若存在则检查类型）
            for key in ["market_cap", "pe_ratio", "return_on_equity"]:
                if key in metrics:
                    print(f"   {key} = {metrics[key]} (type={type(metrics[key])})")
            print("   ✅ get_financial_metrics 测试完成\n")
    except Exception as e:
        print("   ❌ get_financial_metrics 调用失败：", str(e))
        traceback.print_exc()

    # 6) 测试 get_financial_statements（网络依赖 —— akshare）
    try:
        print("6) 测试 get_financial_statements (网络请求) ...")
        t0 = time.time()
        statements = get_financial_statements(symbol)
        t1 = time.time()
        print(f"   请求耗时: {t1 - t0:.2f}s")
        if not statements or not isinstance(statements, list):
            print("   ⚠️ get_financial_statements 返回空或格式不正确")
        else:
            print(f"   返回期间数: {len(statements)} (期数, 每项为 dict)")
            sample_item = statements[0]
            print("   项目 keys:", list(sample_item.keys()))
            for k, v in sample_item.items():
                print(f"     {k}: {v} (type={type(v)})")
            print("   ✅ get_financial_statements 基本测试完成\n")
    except Exception as e:
        print("   ❌ get_financial_statements 调用失败：", str(e))
        traceback.print_exc()

    # 7) 总结与建议
    print("\n=== 测试总结 ===")
    print("注意：第 3/5/6 项依赖 akshare 网络请求；若运行环境无网络或 akshare 数据源临时不可用，可能返回空 DataFrame 或抛出异常。")
    print("若遇到数据源/网络错误，请检查：")
    print("  1) 是否已正确安装 akshare 且版本兼容")
    print("  2) 网络是否可访问外网（部分 akshare 接口需访问第三方站点）")
    print("  3) 传入的股票代码是否正确（示例使用 A 股代码，如 600000）")
    print("\n如需，我可以把上述测试改造成 pytest 风格的单元测试（便于 CI 集成），并替换网络调用为可注入的 mock 接口。")

    print("\n=== api_tool 单元测试结束 ===")
