import os
import sys

from pathlib import Path
import uuid  # Import uuid for run IDs
import traceback
import getpass # For API keys
import chainlit as cl # Import Chainlit
from playwright.async_api import async_playwright
from dotenv import load_dotenv # Import dotenv
from datetime import datetime, timedelta
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
import pprint # For pretty printing the final state
import plotly.graph_objects as go
# --- API Key Setup ---
# Load .env file if it exists
load_dotenv()

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

# (设置您的 API 密钥)
_set_if_undefined("BYTEDANCE_API_KEY")
_set_if_undefined("LANGSMITH_API_KEY")

# --- Agent Imports (Copied from your main.py) ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # 假设 main.py 在 src 目录下
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
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
    from src.agents.intent_recognition import intent_recognition_agent, chitchat_agent

    from src.utils.structured_terminal import print_structured_output
    HAS_STRUCTURED_OUTPUT = True
except ImportError as e:
    print(f"Error importing agents: {e}")
    print("Please ensure 'src' directory is in PYTHONPATH or structured correctly relative to chainlit_app.py")
    # 如果导入失败，我们不能继续，所以在这里退出或设置一个标志
    HAS_STRUCTURED_OUTPUT = False



# ======================================================================================
# 定义工作流 (Copied from your main.py)
# ======================================================================================

# 传入状态定义
workflow = StateGraph(AgentState)

# 添加工作流结构（定义点和边之间的关系）
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




# 1. market_data_tool 获取的数据分别传递给 4 个分析 agent 和 1 个分析新闻分析 agent，进行进一步的分析
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
    "online_sentiment_agent",
    "valuation_analysis_tool",
]
workflow.add_edge(analyst_nodes, "bullish_research_agent")
workflow.add_edge(analyst_nodes, "bearish_research_agent")

# 3. 将多头和空头研究员的观点汇总后输入【辩论室】
workflow.add_edge(["bullish_research_agent", "bearish_research_agent"], "tripartite_judgment_agent")

# 4. 辩论时整合后依次通过【风险管理智能体】和【宏观分析智能体】进行分析
workflow.add_edge("tripartite_judgment_agent", "risk_assessment_tool")

# 5. 将新闻分析和宏观数据分析汇总后传给【资产组合经理】生成报告
workflow.add_edge(["risk_assessment_tool", "macro_news_agent", "macro_market_agent"], "summary_synthesis_agent")

# 6. 终点为生成投资建议的【资产组合经理】
workflow.add_edge("summary_synthesis_agent", END)

app = workflow.compile()

# ======================================================================================
# Chainlit 应用程序定义
# ======================================================================================


@cl.on_chat_start
async def on_chat_start():
    """
    当新聊天会话开始时调用。
    我们在这里设置默认的投资组合。
    """
    await cl.Message(
        content="您好！我们是 SignalAI 智能投研团队。\n\n"
                "请输入您想分析的股票，例如：'我想看看万向钱潮是否值得投资'",
    ).send()



@cl.action_callback("test_html_render")
async def handle_test_html(action):

    # 检查文件是否 *实际* 存在于文件系统上
    base_dir = Path(__file__).parent.parent
    file_name = "000066_20251114_pred_90d.html"
    public_file_path = (
            base_dir / "public" / "output_images_kronos" / file_name
    ).resolve()

    if public_file_path.exists():

        iframe_element = cl.File(
            name=file_name,
            path=str(public_file_path),
            display="inline"  # <---- 关键
        )

        await cl.Message(
            content="#### 📊 渲染 HTML 示例 (Iframe)",
            elements=[iframe_element]
        ).send()
    else:
        await cl.Message(content=f"文件不存在: {public_file_path}").send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    处理用户发送的每条消息。
    """
    # 1. 创建一个消息用于显示 "正在运行" 状态
    msg = cl.Message(content="")
    await msg.send()

    # 2. 从会话和消息中收集运行所需的数据
    user_input = message.content
    run_id = str(uuid.uuid4())

    # 获取当前时间（与 main.py 逻辑相同）
    now_dt = datetime.now()
    yesterday = now_dt - timedelta(days=1)
    end_date_dt = yesterday
    start_date_dt = end_date_dt - timedelta(days=365)

    start_date = start_date_dt.strftime('%Y-%m-%d')
    end_date = end_date_dt.strftime('%Y-%m-%d')

    # 3. 构建初始状态 (与 main.py 逻辑相同)
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "data": {
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

    # 4. 运行工作流
    msg.content = "正在运行分析... 这可能需要一些时间。\n"
    await msg.update()

    try:
        final_state = await cl.make_async(app.invoke)(initial_state)

        # 5. 提取最终的文本结果（无论是闲聊还是分析报告）
        messages = final_state.get("messages", [])
        if not messages:
            result_content = "分析完成，但未找到最终报告。"
        else:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                result_content = last_msg.content
            elif isinstance(last_msg, dict):
                result_content = last_msg.get("content", "分析完成，但未找到最终报告。")
            else:
                result_content = str(last_msg)

        # 6.
        # 检查 agent 返回的意图
        intent = final_state.get("data", {}).get("intent")

        if intent == "stock_analysis":
            # 意图是股票分析：继续渲染和发送图片
            elements_to_send = []
            # 此时 fig_object 是一个 dict (或 None)
            fig_object = final_state.get("data", {}).get("prediction_plotly_fig")
            if fig_object:
                try:
                    if isinstance(fig_object, dict):
                        # 从字典重新构建(反序列化) Figure 对象
                        reconstructed_fig = go.Figure(fig_object)
                    elif isinstance(fig_object, go.Figure):
                        reconstructed_fig = fig_object
                    else:
                        raise TypeError(f"期望的是 dict 或 go.Figure，但收到了 {type(fig_object)}")
                    plotly_element = cl.Plotly(
                        name="股价预测图",
                        figure=reconstructed_fig,
                        display="inline",
                        size="large",
                    )
                    plotly_element.height = 800
                    elements_to_send.append(plotly_element)

                except Exception as e_render:
                    tb = traceback.format_exc()
                    result_content += f"\n\n⚠️ Plotly 对象渲染出错：{e_render}\n```\n{tb}\n```"
            else:
                result_content += "\n\n*(未在 State 中找到 Plotly 对象。)*"


            stock_ticker = final_state.get("data", {}).get("ticker", "")
            current_date_str = datetime.now().strftime("%Y%m%d")
            # 确保 ticker 有效
            if not stock_ticker:
                await cl.Message(content=result_content + "\n\n⚠️ 分析意图已识别，但未找到有效的股票代码。").send()
                return

            # 明确以脚本文件所在目录为基准
            base_dir = Path(__file__).parent.parent
            file_name = f"{stock_ticker}_{current_date_str}_pred_90d.html"
            file_path = (
                    base_dir / "output_images_kronos" / file_name
            ).resolve()
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_path.exists():
                try:
                    # 使用 File (来自顶部的 import)
                    # display="side" (或默认) 会将其显示为附件
                    html_attachment = cl.File(
                        name=file_name,
                        path=str(file_path),
                        display="side"
                    )
                    elements_to_send.append(html_attachment)
                except Exception as e_file:
                    result_content += f"\n\n*(附加 HTML 文件时出错: {e_file})*"
            else:
                result_content += f"\n\n*(未找到 HTML 附件: {file_path})*"

            # 4. 发送最终消息
            if elements_to_send:
                await cl.Message(
                    content=result_content + "\n\n#### 📊 以下是该股票未来90日的预测图：\n\n",
                    elements=elements_to_send
                ).send()
            else:
                await cl.Message(
                    content=result_content + f"\n\n⚠️ 预测完成，但渲染图表和附加文件均失败。"
                ).send()

        else:
            # 意图是 "chitchat" 或其他：只发送文本回复
            await cl.Message(content=result_content).send()

        # 8. 发送完整的状态以供调试
        if HAS_STRUCTURED_OUTPUT:
            state_details = pprint.pformat(final_state, indent=2, width=120)
            await cl.Message(
                content="**完整的最终状态 (调试信息):**",
                elements=[cl.Code(content=state_details, language="python", display="inline")]
            ).send()

        # 9. 删除初始的 "正在运行" 消息
        msg.content = "分析流程已完成。"
        await msg.update()

    except Exception as e:
        tb = traceback.format_exc()
        msg.content = f"运行分析时出错：\n{e}\n```\n{tb}\n```"
        await msg.update()