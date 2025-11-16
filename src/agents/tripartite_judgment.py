from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.openrouter_config import get_chat_completion
from src.utils.logging_config import setup_logger

import json
import ast
import logging

# 获取日志记录器
logger = setup_logger('tripartite_judgment_agent')


def tripartite_judgment_agent(state: AgentState):
    """Facilitates debate between bull and bear researchers to reach a balanced conclusion."""
    show_workflow_status("tripartite_judgment_agent")
    show_reasoning = state["metadata"]["show_reasoning"]

    # 收集所有研究员信息 - 向前兼容设计（添加防御性检查）
    researcher_messages = {}
    for msg in state["messages"]:
        # 添加防御性检查，确保 msg 和 msg.name 不为 None
        if msg is None:
            continue
        if not hasattr(msg, 'name') or msg.name is None:
            continue
        if isinstance(msg.name, str) and msg.name.startswith(("bullish_", "bearish_")):
            researcher_messages[msg.name] = msg
            logger.debug(f"收集到研究员信息: {msg.name}")

    # 确保至少有看多和看空两个研究员
    if "bullish_research_agent" not in researcher_messages or "bearish_research_agent" not in researcher_messages:
        logger.error(
            "缺少必要的研究员数据: bullish_research_agent 或 bearish_research_agent")
        raise ValueError(
            "Missing required bullish_research_agent or bearish_research_agent messages")

    # 处理研究员数据
    researcher_data = {}
    for name, msg in researcher_messages.items():
        # 添加防御性检查，确保 msg.content 不为 None
        if not hasattr(msg, 'content') or msg.content is None:
            logger.warning(f"研究员 {name} 的消息内容为空")
            continue
        try:
            data = json.loads(msg.content)
            logger.debug(f"成功解析 {name} 的 JSON 内容")
        except (json.JSONDecodeError, TypeError):
            try:
                data = ast.literal_eval(msg.content)
                logger.debug(f"通过 ast.literal_eval 解析 {name} 的内容")
            except (ValueError, SyntaxError, TypeError):
                # 如果无法解析内容，跳过此消息
                logger.warning(f"无法解析 {name} 的消息内容，已跳过")
                continue
        researcher_data[name] = data

    # 获取看多和看空研究员数据（为了兼容原有逻辑）
    if "bullish_research_agent" not in researcher_data or "bearish_research_agent" not in researcher_data:
        logger.error("无法解析必要的研究员数据")
        raise ValueError(
            "Could not parse required bullish_research_agent or bearish_research_agent messages")

    bull_thesis = researcher_data["bullish_research_agent"]
    bear_thesis = researcher_data["bearish_research_agent"]

    # 比较置信度级别
    bull_confidence = bull_thesis.get("confidence", 0)
    bear_confidence = bear_thesis.get("confidence", 0)

    # 分析辩论观点
    debate_summary = []
    debate_summary.append("Bullish Arguments:")
    for point in bull_thesis.get("thesis_points", []):
        debate_summary.append(f"+ {point}")

    debate_summary.append("\nBearish Arguments:")
    for point in bear_thesis.get("thesis_points", []):
        debate_summary.append(f"- {point}")

    # 收集所有研究员的论点，准备发给 LLM
    all_perspectives = {}
    for name, data in researcher_data.items():
        perspective = data.get("perspective", name.replace(
            "researcher_", "").replace("_agent", ""))
        all_perspectives[perspective] = {
            "confidence": data.get("confidence", 0),
            "thesis_points": data.get("thesis_points", [])
        }


    # 构建发送给 LLM 的提示
    llm_prompt = """
你是一位专业的金融分析师，请分析以下投资研究员的观点，并给出你的第三方分析:

"""
    for perspective, data in all_perspectives.items():
        llm_prompt += f"\n{perspective.upper()} 观点 (置信度: {data['confidence']}):\n"
        for point in data["thesis_points"]:
            llm_prompt += f"- {point}\n"

    llm_prompt += """
请提供以下格式的 JSON 回复:
{
    "analysis": "你的详细分析，评估各方观点的优劣，并指出你认为最有说服力的论点",
    "score": 0.5,  // 你的评分，从 -1.0(极度看空) 到 1.0(极度看多)，0 表示中性
    "reasoning": "你给出这个评分的简要理由"
}

务必确保你的回复是有效的 JSON 格式，且包含上述所有字段。回复使用中文。
"""

    # 调用 LLM 获取第三方观点
    llm_response = None
    llm_analysis = None
    llm_score = 0  # 默认为中性
    try:
        logger.info("调用 LLM 对看空和看多的结论进行第三方分析...")
        messages = [
            {"role": "system", "content": "你是一个专业的金融分析师，请用中文阐述你的分析过程与结果。"},
            {"role": "user", "content": llm_prompt}
        ]

        # 使用log_llm_interaction装饰器记录LLM交互
        llm_response = get_chat_completion(messages)


        # 解析 LLM 返回的 JSON
        if llm_response:
            try:
                # 尝试提取 JSON 部分
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = llm_response[json_start:json_end]
                    llm_analysis = json.loads(json_str)
                    llm_score = float(llm_analysis.get("score", 0))
                    # 确保分数在有效范围内
                    llm_score = max(min(llm_score, 1.0), -1.0)
                    # logger.info(f"成功解析 LLM 回复，评分: {llm_score}")
                    logger.debug(
                        f"LLM 分析内容: {llm_analysis.get('analysis', '未提供分析')[:100]}...")
            except Exception as e:
                # 如果解析失败，记录错误并使用默认值
                logger.error(f"解析 LLM 回复失败: {e}")
                llm_analysis = {"analysis": "Failed to parse LLM response",
                                "score": 0, "reasoning": "Parsing error"}
    except Exception as e:
        logger.error(f"调用 LLM 失败: {e}")
        llm_analysis = {"analysis": "LLM API call failed",
                        "score": 0, "reasoning": "API error"}

    # 计算混合置信度差异
    confidence_diff = bull_confidence - bear_confidence

    # 默认 LLM 权重为 30%
    llm_weight = 0.3

    # 将 LLM 评分（-1 到 1范围）转换为与 confidence_diff 相同的比例
    # 计算混合置信度差异
    mixed_confidence_diff = (1 - llm_weight) * \
        confidence_diff + llm_weight * llm_score

    # logger.info(
    #     f"计算混合置信度差异: 原始差异={confidence_diff:.4f}, LLM评分={llm_score:.4f}, 混合差异={mixed_confidence_diff:.4f}")

    # 基于混合置信度差异确定最终建议
    if abs(mixed_confidence_diff) < 0.1:  # 接近争论
        final_signal = "neutral"
        reasoning = "Balanced debate with strong arguments on both sides"
        confidence = max(bull_confidence, bear_confidence)
    elif mixed_confidence_diff > 0:  # 看多胜出
        final_signal = "bullish"
        reasoning = "Bullish arguments more convincing"
        confidence = bull_confidence
    else:  # 看空胜出
        final_signal = "bearish"
        reasoning = "Bearish arguments more convincing"
        confidence = bear_confidence

    # logger.info(f"最终投资信号: {final_signal}, 置信度: {confidence}")

    # 构建返回消息，包含 LLM 分析
    message_content = {
        "signal": final_signal,
        "confidence": confidence,
        "bull_confidence": bull_confidence,
        "bear_confidence": bear_confidence,
        "confidence_diff": confidence_diff,
        "llm_score": llm_score if llm_analysis else None,
        "llm_analysis": llm_analysis["analysis"] if llm_analysis and "analysis" in llm_analysis else None,
        "llm_reasoning": llm_analysis["reasoning"] if llm_analysis and "reasoning" in llm_analysis else None,
        "mixed_confidence_diff": mixed_confidence_diff,
        "debate_summary": debate_summary,
        "reasoning": reasoning
    }

    message = HumanMessage(
        content=json.dumps(message_content, ensure_ascii=False),
        name="tripartite_judgment_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "tripartite_judgment_agent")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("tripartite_judgment_agent", "completed")
    return {
        "messages": state["messages"] + [message],
        "data": {
            **state["data"],
            "debate_analysis": message_content
        },
        "metadata": state["metadata"],
    }
