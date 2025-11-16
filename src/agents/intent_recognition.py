import sys, os
import json
import re
from datetime import datetime, timedelta
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # 假设 main.py 在 src 目录下
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.logging_config import setup_logger
from src.tools.openrouter_config import get_chat_completion, llm # 调用大模型
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status # 状态记录
from langgraph.types import interrupt, Command, RetryPolicy
# 初始化 logger
logger = setup_logger('intent_recognition_agent')

class IntentRecognition:
    intent: Literal["stock_analysis", "chitchat"]
    ticker: str | None





def intent_recognition_agent(state: AgentState) -> Command[Literal["market_data_tool", "chitchat_agent"]]:

    show_workflow_status("Intent Recognition Agent")
    show_reasoning = state["metadata"].get("show_reasoning", False)

    first_message = state["messages"][0]
    user_input = first_message.content
    data = state["data"]

    system_message = {
        "role": "system",
        "content": """
        你是一个A股投资意图识别助手。
        你的任务是：
        1. 从用户输入中判断用户是否在讨论具体的A股股票；
        2. 如果有，则提取：
         - 股票对应的6位数字代码，
         - 股票中文名称(xxx股票)；
        3. 如果没有，则判断该输入属于闲聊、问候或其他非投资分析意图。

        输出格式要求：
        仅返回一个JSON对象，包含两个字段：
        - intent: "stock_analysis" 或 "chitchat"
        - ticker: 若存在股票则为6位数字字符串，否则为 null

        示例：
        用户输入："帮我看看贵州茅台"
        输出：{"intent": "stock_analysis", "ticker": "600519", "stock_name": "贵州茅台股票"}

        用户输入："你好啊，最近市场热不热？"
        输出：{"intent": "chitchat", "ticker": null,"stock_name": null}
        """
    }
    user_message = {
        "role": "user",
        "content": f"请根据以下文字判断意图：\n\n{user_input}\n\n请严格返回JSON格式结果"
    }

    intent_result = None
    max_attempts = 3
    for _ in range(max_attempts):
        try:
            structured_llm = llm.with_structured_output(IntentRecognition)  
            result = structured_llm.invoke([system_message, user_message])
            # 尝试解析结果
            parsed = result  # type: ignore
            # 验证结果
            if not isinstance(parsed, dict):
                raise ValueError("非字典结构")
            if "intent" not in parsed or "ticker" not in parsed:
                raise ValueError("缺少必要字段")
            if parsed["intent"] not in ["stock_analysis", "chitchat"]:
                raise ValueError("intent 字段无效")

            intent_result = parsed
            break
        except Exception as e:
            logger.warning(f"解析意图识别结果失败: {e}")
            continue

     # === 失败回退逻辑 ===
    if not intent_result:
        intent_result = {"intent": "chitchat", "ticker": None}

    # === 路由逻辑 ===
    goto = (
        "market_data_tool"
        if intent_result["intent"] == "stock_analysis" and intent_result["ticker"]
        else "chitchat_agent"
    )

    show_workflow_status("Intent Recognition Agent", f"completed → {goto}")

    # === 更新状态 ===
    return Command(
        update={
            "data": {
                **data,
                "intent": intent_result["intent"],
                "ticker": intent_result["ticker"],
                "stock_name": intent_result.get("stock_name","无股票分析意图")
            }
        },
        goto=goto,
    ) # type: ignore

from dataclasses import dataclass

@dataclass
class ChitchatResponse:
    response: str


def chitchat_agent(state: AgentState) -> ChitchatResponse:
    """
    LangGraph 智能体：处理闲聊意图，生成回复。
    """
    show_workflow_status("Chitchat Agent")
    show_reasoning = state["metadata"].get("show_reasoning", False)

    first_message = state["messages"][0]
    user_input = first_message.content

    system_message = {
        "role": "system",
        "content": """
        你是一个友好的A股投资助理。当用户的输入不涉及具体股票分析时，你需要进行闲聊、问候或提供一般性的信息。
        请确保你的回复简洁、有帮助且符合投资助理的角色。
        """
    }
    user_message = {
        "role": "user",
        "content": f"请根据以下文字进行回复：\n\n{user_input}"
    }

    result = get_chat_completion([system_message, user_message])
    if show_reasoning:
        show_agent_reasoning(result, "Chitchat Agent")

    show_workflow_status("Chitchat Agent", "completed")

    return {
        "messages": state["messages"] + [AIMessage(content=result)] # type: ignore
    }

