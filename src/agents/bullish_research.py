import json
import ast
import logging
from typing import Literal # 导入
from langchain_core.messages import HumanMessage, AIMessage # 导入 AIMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.openrouter_config import get_chat_completion, llm # 假设 llm 也来自这里

# 设置日志
logger = logging.getLogger("bullish_research_agent")

class BullishAnalysisOutput:
    """定义看多研究员的结构化输出"""
    perspective: Literal["bullish"]
    confidence: float
    thesis_points: list[str]
    reasoning: str

def bullish_research_agent(state: AgentState):
    """
    (已更新: LLM-based)
    从看多角度分析所有信号 (包括预测数据)，并生成乐观的投资论点。
    """
    show_workflow_status("bullish_research_agent")
    show_reasoning = state["metadata"]["show_reasoning"]
    
    # -----------------------------------------------------------------
    # 1. 获取上游分析师数据
    # -----------------------------------------------------------------
    try:
        technical_message = next(
            msg for msg in state["messages"] if msg.name == "technical_analysis_tool")
        fundamentals_message = next(
            msg for msg in state["messages"] if msg.name == "fundamentals_analysis_tool")
        valuation_message = next(
            msg for msg in state["messages"] if msg.name == "valuation_analysis_tool")
            
        fundamental_signals = json.loads(fundamentals_message.content) # type: ignore
        technical_signals = json.loads(technical_message.content) # type: ignore
        valuation_signals = json.loads(valuation_message.content) # type: ignore
        
    except Exception as e:
        logger.warning(f"解析上游分析师数据失败: {e}，尝试 ast.literal_eval...")
        try:
            fundamental_signals = ast.literal_eval(fundamentals_message.content) # type: ignore
            technical_signals = ast.literal_eval(technical_message.content) # type: ignore
            valuation_signals = ast.literal_eval(valuation_message.content) # type: ignore
        except Exception as e_ast:
            msg = f"❌ 无法解析上游分析师信号: {e_ast}"
            logger.error(msg)
            state["messages"].append(AIMessage(content=msg, name="bullish_research_agent")) # type: ignore
            return state

    # -----------------------------------------------------------------
    # 2. 获取 AI 预测数据
    # -----------------------------------------------------------------
    prediction_analysis = state["data"].get("prediction_analysis")
    
    if prediction_analysis:
        # 我们只提取最关键的摘要信息，而不是整个字典，以节省 Token
        summary_table = prediction_analysis.get("summary_table", {})
        text_report = prediction_analysis.get("text_report", "No text report available.")
        
        prediction_context = f"""
        5.  **AI模型时间序列预测 (Kronos Forecast)**:
            -   **摘要表 (Summary Table)**: {json.dumps(summary_table)}
            -   **文本报告 (Text Report)**: "{text_report}"
        """
    else:
        prediction_context = "5.  **AI模型时间序列预测 (Kronos Forecast)**: 未提供数据。"
        
    # -----------------------------------------------------------------
    # 3. 定义 LLM Prompt
    # -----------------------------------------------------------------
    
    system_prompt = f"""
    你是一个专业的、持极度乐观态度的股票分析师（多方研究员）。
    你的唯一任务是审查所有输入的数据，并**只从看多（Bullish）的角度**进行解读。

    你的职责：
    1.  **放大正面信号**：如果数据（如基本面、估值）是正面的，你要大力赞扬它们是强劲的增长动力。
    2.  **淡化负面信号**：如果数据（如技术、估值）是负面的，你必须将其解读为“暂时的回调”、“市场非理性恐慌”、“绝佳的买入机会”或“空头陷阱”。
    3.  **寻找预测中的机会**：从 `Kronos Forecast` 数据中找出机会点。
        -   如果它建议 `reduce` (减仓)，你必须将其解读为“AI模型过于保守，未能量化真正的上行潜力”。
        -   如果它预测高收益 (如 `annualized_return: 101.04%`)，你要强调这是“爆炸性增长”的信号，并淡化波动 (`annualized_vol`)，称之为“高增长的正常代价”。
        -   指出宽的置信区间 (CI) 意味着“巨大的上行空间”（例如，引用 `ci_95_upper` 作为潜在目标）。
    4.  **提供结构化输出**：你必须严格按照指定的JSON格式返回你的分析。
    """

    user_prompt = f"""
    请分析以下关于某只股票的5个综合数据源，并提供你的**看多分析报告**。

    **输入数据源：**

    1.  **基本面分析 (Fundamentals)**: {json.dumps(fundamental_signals)}
    2.  **技术分析 (Technical)**: {json.dumps(technical_signals)}
    3.  **估值分析 (Valuation)**: {json.dumps(valuation_signals)}
    {prediction_context}

    **你的任务：**
    基于上述所有信息，生成一份看多报告。记住，即使所有信号都看跌，你也要找出潜在的“超卖反弹”、“错杀的机会”或“黎明前的黑暗”。

    **输出格式 (必须严格遵守此JSON结构)：**
    {{
      "perspective": "bullish",
      "confidence": <一个0.0到1.0之间的浮点数，表示你对看多观点的信心>,
      "thesis_points": [
        "<看多论点1，基于一个数据点>",
        "<看多论点2，基于另一个数据点>",
        "<看多论点3，(重点) 综合 AI 预测数据中的机会点>"
      ],
      "reasoning": "<一个字符串，总结你为什么得出这些看多结论>"
    }}
    """
    
    # -----------------------------------------------------------------
    # 4. 调用 LLM 并解析结果
    # -----------------------------------------------------------------
    try:
        # 调用大模型，并指定结构化输出
        structured_llm = llm.with_structured_output(BullishAnalysisOutput)
        structured_llm_response = structured_llm.invoke(input=[system_prompt, user_prompt])

        if show_reasoning:
            show_agent_reasoning(f"LLM Raw Response:\n{structured_llm_response}", "bullish_research_agent")

        # 确保 LLM 遵守了指令
        structured_llm_response["perspective"] = "bullish"

    except Exception as e:
        logger.error(f"bullish_research_agent 失败或解析JSON错误: {e}")
        # (回退机制) 如果 LLM 失败，返回一个通用的错误信息
        structured_llm_response = {
            "perspective": "bullish",
            "confidence": 0.5,
            "thesis_points": [
                f"LLM 分析失败: {str(e)}",
                "由于系统分析模块出现故障，无法评估机会。",
                "建议保持中立观察。"
            ],
            "reasoning": "LLM analysis failed or returned invalid JSON. Defaulting to neutral."
        }

    # -----------------------------------------------------------------
    # 5. 封装并返回状态
    # -----------------------------------------------------------------
    message = HumanMessage(
        content=json.dumps(structured_llm_response),
        name="bullish_research_agent",
    )

    if show_reasoning:
        show_agent_reasoning(structured_llm_response, "bullish_research_agent") # <-- 修改 Agent 名称
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = structured_llm_response

    show_workflow_status("bullish_research_agent", "completed") # <-- 修改 Agent 名称
    
    # 返回 state 时要确保 state["messages"] 确实被更新了
    current_messages = state.get("messages", [])
    current_messages.append(message) # type: ignore
    
    return {
        "messages": current_messages,
        "data": state["data"],
        "metadata": state["metadata"],
    }