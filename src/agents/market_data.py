import sys, os

# 自动添加项目根目录到 sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from langchain_core.messages import HumanMessage


from .state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.api import get_financial_metrics, get_financial_statements, get_price_history
from src.utils.logging_config import setup_logger


from datetime import datetime, timedelta
import pandas as pd

# 设置日志记录
logger = setup_logger('market_data_tool')


def market_data_tool(state: AgentState)-> dict:
    """Responsible for gathering and preprocessing market data"""
    # 1. 前置准备
    show_workflow_status("market_data_tool") # 日志提示开始工作
    show_reasoning = state["metadata"]["show_reasoning"] # 是否需要显示推理过程
    # 2. 获取：状态数据信息 & 时间信息（必须是昨天）
    messages = state["messages"]
    data = state["data"]
    current_date = datetime.now()
    yesterday = current_date - timedelta(days=1)
    end_date = data["end_date"] or yesterday.strftime('%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    if end_date_obj > yesterday:
        end_date = yesterday.strftime('%Y-%m-%d')
        end_date_obj = yesterday
    if not data["start_date"]:
        start_date = end_date_obj - timedelta(days=365)  # 默认获取一年的数据
        start_date = start_date.strftime('%Y-%m-%d')
    else:
        start_date = data["start_date"]
    # 3. 确定需要分析的目标对象（股票代码）
    ticker = data["ticker"]

    # 4. 获取核心数据
    ## 股票价格数据
    prices_df = get_price_history(ticker, start_date, end_date)
    if prices_df is None or prices_df.empty:
        logger.warning(f"警告：无法获取{ticker}的价格数据，将使用空数据继续")
        # 收盘 & 开盘 & 最高 & 最低 & 成交量 （抛弃字段：成交额 & 振幅 & 涨跌幅 & 涨跌额 & 换手率）
        prices_df = pd.DataFrame(columns=['close', 'open', 'high', 'low', 'volume'])
    if not isinstance(prices_df, pd.DataFrame): # 确保数据格式正确
        prices_df = pd.DataFrame(
            columns=['close', 'open', 'high', 'low', 'volume'])
    prices_dict = prices_df.to_dict('records') # 转换价格数据为字典格式
    ## 财务指标数据
    try:
        financial_metrics = get_financial_metrics(ticker)
    except Exception as e:
        logger.error(f"获取财务指标失败: {str(e)}")
        financial_metrics = {}

    ## 财务报表数据
    try:
        financial_line_items = get_financial_statements(ticker)
    except Exception as e:
        logger.error(f"获取财务报表失败: {str(e)}")
        financial_line_items = {}

    # ## 市场数据单独取出来
    try:
        market_data ={
        "market_cap":  financial_metrics[0]["market_cap"],
        "float_market_cap":  financial_metrics[0]["float_market_cap"],
        "general_capital":  financial_metrics[0]["general_capital"],
        "float_capital":  financial_metrics[0]["float_capital"],
        }
    except Exception as e:
        logger.error(f"获取市场数据失败: {str(e)}")
        market_data = {"market_cap": 0}

    # 5. 保存推理信息到metadata
    market_data_summary = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "data_collected": {
            "price_history": len(prices_dict) > 0,
            "financial_metrics": len(financial_metrics) > 0,
            "financial_statements": len(financial_line_items) > 0,
        },
        "summary": f"为{ticker}收集了从{start_date}到{end_date}的市场数据，包括价格历史、财务指标和市场信息"
    }

    # 6. 输出推理日志
    if show_reasoning:
        show_agent_reasoning(market_data_summary, "market_data_tool")
        state["metadata"]["agent_reasoning"] = market_data_summary

    # 7. 结束状态
    show_workflow_status("market_data_tool", "completed")

    return {
        "messages": messages,
        "data": {
            **data,
            "prices": prices_dict,
            "start_date": start_date,
            "end_date": end_date,
            "financial_metrics": financial_metrics,
            "financial_line_items": financial_line_items,
            "market_cap": market_data.get("market_cap", 0),
            "market_data": market_data,
        },
        "metadata": state["metadata"],
    }

