import json
import ast
import logging
from typing import Literal # 导入日志
from langchain_core.messages import HumanMessage, AIMessage # 导入 AIMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.openrouter_config import get_chat_completion,llm

# 设置日志
logger = logging.getLogger("bearish_research_agent")


class BearishAnalysisOutput:
    perspective: Literal["bearish"]
    confidence: float
    thesis_points: list[str]
    reasoning: str


def bearish_research_agent(state: AgentState):
    """
    从看空角度分析所有信号 (包括预测数据)，并生成谨慎的投资论点。
    """
    show_workflow_status("bearish_research_agent")
    show_reasoning = state["metadata"]["show_reasoning"]
    
    # -----------------------------------------------------------------
    # 1. (保留) 获取上游分析师数据
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
            state["messages"].append(AIMessage(content=msg, name="bearish_research_agent")) # type: ignore
            return state

    # -----------------------------------------------------------------
    # 2.  获取 AI 预测数据
    # -----------------------------------------------------------------
    prediction_analysis = state["data"].get("prediction_analysis")
    prediction_context = ""
    
    if prediction_analysis:
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
    你是一个专业的、持极度悲观态度的股票分析师（空方研究员）。
    你的唯一任务是审查所有输入的数据，并**只从看空（Bearish）的角度**进行解读。

    你的职责：
    1.  **放大负面信号**：如果数据（如技术、估值）是负面的，你要猛烈抨击它们。
    2.  **淡化正面信号**：如果数据（如基本面、估值）是正面的，你必须将其解读为“不可持续的泡沫”、“陷阱”、“市场非理性”或“诱多的陷阱”。
    3.  **寻找预测中的风险**：从 `Kronos Forecast` 数据中找出风险点。
        -   如果它建议 `reduce` (减仓) 或 `defensive` (防御)，你要强调这是 AI 发出的明确卖出信号。
        -   如果它预测高收益 (如 `annualized_return: 101.04%`)，你必须指出这伴随着极高的波动 (`annualized_vol: 51.79%`) 和巨大的回撤 (`max_drawdown: -13.30%`)，称之为“高风险的赌博”。
        -   指出宽的置信区间 (CI) 意味着“极度的不确定性”。
    4.  **提供结构化输出**：你必须严格按照指定的JSON格式返回你的分析。
    """

    user_prompt = f"""
    请分析以下关于某只股票的5个综合数据源，并提供你的**看空分析报告**。

    **输入数据源：**

    1.  **基本面分析 (Fundamentals)**: {json.dumps(fundamental_signals)}
    2.  **技术分析 (Technical)**: {json.dumps(technical_signals)}
    3.  **估值分析 (Valuation)**: {json.dumps(valuation_signals)}
    {prediction_context}

    **你的任务：**
    基于上述所有信息，生成一份看空报告。记住，即使所有信号都看涨，你也要找出潜在的“非理性繁荣”或“最后的疯狂”。

    **输出格式 (必须严格遵守此JSON结构)：**
    {{
      "perspective": "bearish",
      "confidence": <一个0.0到1.0之间的浮点数，表示你对看空观点的信心>,
      "thesis_points": [
        "<看空论点1，基于一个数据点>",
        "<看空论点2，基于另一个数据点>",
        "<看空论点3，(重点) 综合 AI 预测数据中的风险点>"
      ],
      "reasoning": "<一个字符串，总结你为什么得出这些看空结论>"
    }}
    """
    
    # -----------------------------------------------------------------
    # 4. 调用 LLM 并解析结果
    # -----------------------------------------------------------------
    try:
        structured_llm = llm.with_structured_output(BearishAnalysisOutput)
        structured_llm_response = structured_llm.invoke(input=[system_prompt,user_prompt])

        if show_reasoning:
            show_agent_reasoning(f"LLM Raw Response:\n{structured_llm_response}", "bearish_research_agent")

        # 确保 LLM 遵守了指令
        structured_llm_response["perspective"] = "bearish"
    except Exception as e:
        logger.error(f"bearish_research_agent 失败或解析JSON错误: {e}")
        # (回退机制) 如果 LLM 失败，返回一个通用的错误信息
        structured_llm_response = {
            "perspective": "bearish",
            "confidence": 0.5,
            "thesis_points": [
                f"LLM 分析失败: {str(e)}",
                "由于系统分析模块出现故障，无法评估风险。",
                "建议采取极度谨慎的防御姿态。"
            ],
            "reasoning": "LLM alysis failed or returned invalid JSON. Defaulting to maximum caution."
        }

    # -----------------------------------------------------------------
    # 5. 封装并返回状态
    # -----------------------------------------------------------------
    message = HumanMessage(
        content=json.dumps(structured_llm_response),
        name="bearish_research_agent",
    )

    if show_reasoning:
        show_agent_reasoning(structured_llm_response, "bearish_research_agent")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = structured_llm_response

    show_workflow_status("bearish_research_agen", "completed")
    
    # 返回 state 时要确保 state["messages"] 确实被更新了
    current_messages = state.get("messages", [])
    current_messages.append(message) # type: ignore
    
    return {
        "messages": current_messages,
        "data": state["data"],
        "metadata": state["metadata"],
    }