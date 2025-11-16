import os
import sys
import argparse
import uuid  # Import uuid for run IDs
import threading  # Import threading for background task
# import uvicorn  # Import uvicorn to run FastAPI

from datetime import datetime, timedelta
# Removed START as it's implicit with set_entry_point
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
# import pandas as pd
# import akshare as ak

# --- Agent Imports ---
import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # 假设 main.py 在 src 目录下
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.agents.valuation import valuation_analysis_tool
from src.agents.state import AgentState
from src.agents.online_sentiment import online_sentiment_agent
from src.agents.risk_assessment import risk_assessment_tool
from src.agents.technicals import technical_analysis_tool
from src.agents.stock_forecast import stock_forecast_tool
from src.agents.summary_synthesis import summary_synthesis_agent
from src.agents.market_data import market_data_tool
from src.agents.fundamentals import fundamentals_analysis_tool
from src.agents.bullish_research import bullish_research_agent
from src.agents.bearish_research import bearish_research_agent
from src.agents.tripartite_judgment import tripartite_judgment_agent
from src.agents.macro_market import macro_market_agent
from src.agents.macro_news import macro_news_agent
from src.agents.intent_recognition import intent_recognition_agent,chitchat_agent

try:
    from src.utils.structured_terminal import print_structured_output
    HAS_STRUCTURED_OUTPUT = True
except ImportError:
    HAS_STRUCTURED_OUTPUT = False

# ======================================================================================

# 工作流运行函数

def run_hedge_fund(
        run_id: str,
        user_input: str,
        start_date: str,
        end_date: str):
    print(f"--- 开始投资策略分析 Run ID: {run_id} ---")

    messages = [HumanMessage(content=user_input)]

    initial_state = {
        "messages": messages,
        "data": {
            # "portfolio": portfolio,
            "start_date": start_date,
            "end_date": end_date,
            "num_of_news": 100,
        },
        "metadata": {
            "show_reasoning": True,
            "run_id": run_id,
            "show_summary": True
        }
    }


    final_state = app.invoke(initial_state) # type: ignore # 将初始状态传入工作流，触发整个流程运行，返回运行涉及的所有信息
    print(f"--- 投资策略分析完成 Run ID: {run_id} ---")

    if HAS_STRUCTURED_OUTPUT:
        print_structured_output(final_state) # type: ignore

    return final_state["messages"][-1]["content"]

# ======================================================================================

# 定义工作流
## 传入状态定义
workflow = StateGraph(AgentState)
## 添加工作流结构（定义点和边之间的关系）
workflow.add_node("intent_recognition_agent", intent_recognition_agent)
workflow.add_node("market_data_tool", market_data_tool)
workflow.add_node("technical_analysis_tool", technical_analysis_tool)
workflow.add_node("stock_forecast_tool", stock_forecast_tool)
workflow.add_node("fundamentals_analysis_tool", fundamentals_analysis_tool)
workflow.add_node("online_sentiment_agent", online_sentiment_agent)
workflow.add_node("valuation_analysis_tool", valuation_analysis_tool)
workflow.add_node("macro_news_agent", macro_news_agent)
workflow.add_node("bullish_research_agent", bullish_research_agent)
workflow.add_node("bearish_research_agent", bearish_research_agent)
workflow.add_node("tripartite_judgment_agent", tripartite_judgment_agent)
workflow.add_node("risk_assessment_tool", risk_assessment_tool)
workflow.add_node("macro_market_agent", macro_market_agent)
workflow.add_node("summary_synthesis_agent", summary_synthesis_agent)

workflow.add_node("chitchat_agent", chitchat_agent)
# ==================== 边定义 ====================

workflow.set_entry_point("intent_recognition_agent")


# 1. market_data_tool 获取的数据分别传递给 agent
workflow.add_edge("market_data_tool", "technical_analysis_tool")
workflow.add_edge("market_data_tool", "stock_forecast_tool")
workflow.add_edge("market_data_tool", "fundamentals_analysis_tool")
workflow.add_edge("market_data_tool", "online_sentiment_agent")
workflow.add_edge("market_data_tool", "valuation_analysis_tool")
workflow.add_edge("market_data_tool", "macro_news_agent")
workflow.add_edge("market_data_tool", "macro_market_agent")


# 2. 将4个初步分析计算结果汇总后，分别传递给【多头研究员】和【空头研究员】
analyst_nodes = [
    "technical_analysis_tool",
    "stock_forecast_tool",
    "fundamentals_analysis_tool",
    "valuation_analysis_tool",
]
workflow.add_edge(analyst_nodes, "bullish_research_agent")
workflow.add_edge(analyst_nodes, "bearish_research_agent")
workflow.add_edge(["bullish_research_agent", "bearish_research_agent"], "tripartite_judgment_agent")
workflow.add_edge("tripartite_judgment_agent", "risk_assessment_tool")

workflow.add_edge(["risk_assessment_tool", "macro_news_agent","macro_market_agent", "online_sentiment_agent"], "summary_synthesis_agent")

workflow.add_edge("summary_synthesis_agent", END)

# 将工作流转换为可执行的程序
app = workflow.compile()

# ======================================================================================

import getpass
import uuid
# --- Main Execution Block ---
if __name__ == "__main__":
    def _set_if_undefined(var: str):
        if not os.environ.get(var):
            os.environ[var] = getpass.getpass(f"Please provide your {var}")


    # (设置您的 API 密钥)
    _set_if_undefined("BYTEDANCE_API_KEY")
    _set_if_undefined("LANGSMITH_API_KEY")
    class Args:
        def __init__(self):
            self.user_input = "我想看看万向钱潮是否值得投资"
            # self.initial_capital = 1000000.0
            # self.initial_position = 1000
    args = Args()

    # 获取当前时间（分析基于前一天时间进行，保证end_data的数据具有完整性，start_data默认为一年前）
    current_date = datetime.now()
    yesterday = current_date - timedelta(days=1)
    end_date = yesterday
    start_date = end_date - timedelta(days=365)


    # 运行函数
    result = run_hedge_fund(
        run_id = str(uuid.uuid4()),
        user_input = args.user_input,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
        # portfolio=portfolio
    )
    print("\nFinal Result:")
    print(result)
