from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import json
import re
from src.utils.logging_config import setup_logger

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.openrouter_config import get_chat_completion
import plotly.graph_objects as go
# 初始化 logger
logger = setup_logger('summary_synthesis_agent')

# (可以放在 summary_synthesis.py 的顶部)
import numpy as np
import collections.abc  # 用于更稳健的 dict/list 检查



import numpy as np
import collections.abc
import plotly.graph_objects as go



##### summary_synthesis_agent #####
def get_latest_message_by_name(messages: list, name: str):
    for msg in reversed(messages):
        if msg.name == name:
            return msg
    logger.warning(
        f"Message from agent '{name}' not found in summary_synthesis_agent.")
    # Return a dummy message object or raise an error, depending on desired handling
    # For now, returning a dummy message to avoid crashing, but content will be None.
    return HumanMessage(content=json.dumps({"signal": "error", "details": f"Message from {name} not found"}), name=name)

def message_to_dict(msg):
    """将 LangChain 的 HumanMessage/AIMessage 转换为可序列化字典"""
    if hasattr(msg, "content"):
        return {
            "type": msg.__class__.__name__,
            "name": getattr(msg, "name", None),
            "content": msg.content,
            "additional_kwargs": getattr(msg, "additional_kwargs", {}),
        }
    elif isinstance(msg, dict):
        return msg
    return str(msg)

def parse_json_signal(signal_str):
    if not signal_str:
        return {}  # 空字符串返回空字典
    try:
        return json.loads(signal_str)  # 解析 JSON 字符串为字典
    except json.JSONDecodeError:
        return {}  # 解析失败也返回空字典


def format_decision(stock_pred_result: str, agent_signals: dict, market_wide_news_summary: str = "未提供") -> dict:
    fundamental_signal = agent_signals.get("fundamental_signal")
    valuation_signal = agent_signals.get("valuation_signal")
    technical_signal =  agent_signals.get("technical_signal")
    sentiment_signal =  agent_signals.get("sentiment_signal")
    risk_signal = agent_signals.get("risk_signal")
    general_macro_signal = agent_signals.get("general_macro_signal")
    market_wide_news_signal = agent_signals.get("market_wide_news_signal")

    def signal_to_chinese(signal_data):
        # 如果是字符串，直接处理
        if isinstance(signal_data, str):
            if signal_data == "bullish":
                return "看多"
            elif signal_data == "bearish":
                return "看空"
            else:
                return "中性"
        # 如果是字典，按原逻辑处理
        elif isinstance(signal_data, dict):
            if signal_data.get("signal") == "bullish":
                return "看多"
            elif signal_data.get("signal") == "bearish":
                return "看空"
            else:
                return "中性"
        # 其他类型默认中性
        else:
            return "中性"

    detailed_analysis = f"""
## 投资分析报告

### 一、投资建议 🎯
操作建议: 【操作建议待补充，从以下3个选项中选一个（1）买入/增持 💹、（2）卖出/减仓 📉 (若暂未持股，则继续保持不持有状态，不建议买入该股票)、（3）继续持有/场外观望 ➡️】

### 二、股票情况分析 💸

#### 1. 基本面分析 (信号: {signal_to_chinese(fundamental_signal)})
\n
##### (1) 相关维度数据
- 盈利能力: {fundamental_signal.get('reasoning', {}).get('profitability_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}
- 增长情况: {fundamental_signal.get('reasoning', {}).get('growth_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}
- 财务健康: {fundamental_signal.get('reasoning', {}).get('financial_health_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}
- 估值水平: {fundamental_signal.get('reasoning', {}).get('price_ratios_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}

##### (2) 基本面情况分析
【基本面情况分析待补充】

#### 2. 估值分析 (信号: {signal_to_chinese(valuation_signal)})
\n
##### (1) 相关维度数据
- DCF估值: {valuation_signal.get('reasoning', {}).get('dcf_analysis', {}).get('details', '无数据') if valuation_signal else '无数据'}
- 所有者收益法: {valuation_signal.get('reasoning', {}).get('owner_earnings_analysis', {}).get('details', '无数据') if valuation_signal else '无数据'}
- 股票走势分析: {stock_pred_result}

##### (2) 估值情况分析
【估值情况分析待补充】

#### 3. 技术分析 (信号: {signal_to_chinese(technical_signal)})
\n
##### (1) 相关维度数据
- 趋势跟踪: ADX={(technical_signal.get('strategy_signals', {}).get('trend_following', {}).get('metrics', {}).get('adx', 0.0) if technical_signal else 0.0) :.2f}
- 均值回归: RSI(14)={(technical_signal.get('strategy_signals', {}).get('mean_reversion', {}).get('metrics', {}).get('rsi_14', 0.0) if technical_signal else 0.0) :.2f}
- 波动性: {(technical_signal.get('strategy_signals', {}).get('volatility', {}).get('metrics', {}).get('historical_volatility', 0.0) if technical_signal else 0.0) :.2%}
- 动量指标:
  1月动量={(technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_1m', 0.0) if technical_signal else 0.0) :.2%}
  3月动量={(technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_3m', 0.0) if technical_signal else 0.0) :.2%}
  6月动量={(technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_6m', 0.0) if technical_signal else 0.0) :.2%}

##### (2) 技术面情况分析
【技术面情况分析待补充】

#### 4. 网络情绪分析 (信号: {sentiment_signal.get('sentiment_signal', '中性') if sentiment_signal else '中性'} & 对股票影响信号: {sentiment_signal.get('sentiment_impact', '中性') if sentiment_signal else '中性'})
\n
##### (1) 关键情绪因素
{'、'.join(sentiment_signal.get('key_factors', ['无关键影响因素分析'])) if isinstance(sentiment_signal.get('key_factors'), list) else sentiment_signal.get('reasoning', '无关键影响因素分析')}

##### (2) 网络情绪对股票影响分析
{'; '.join(sentiment_signal.get('reasoning', ['无详细分析']) if sentiment_signal else ['无详细分析'])}

#### 5. 宏观环境分析
\n
##### (1) 经济环境角度
- 信号: {signal_to_chinese(general_macro_signal.get('macro_environment', '无数据') if general_macro_signal else '无数据')} & 对股票影响信号: {signal_to_chinese(general_macro_signal.get('impact_on_stock', '无数据') if general_macro_signal else '无数据')}
- 关键因素: {';'.join(general_macro_signal.get('key_factors', ['无数据']) if general_macro_signal else ['无数据'])}
- 关键因素分析: 【经济环境角度关键因素分析待补充】

##### (2) 大盘新闻角度
{market_wide_news_signal.get('reasoning', market_wide_news_summary) if market_wide_news_signal else market_wide_news_summary}

#### 6. 风险评估 (市场风险指数: {risk_signal.get('risk_score', '无数据') if risk_signal else '无数据'}/10)
\n
##### (1) 风险维度指标
- 波动率: {(risk_signal.get('risk_metrics', {}).get('volatility', 0.0) * 100 if risk_signal else 0.0) :.1f}%
- 最大回撤: {(risk_signal.get('risk_metrics', {}).get('max_drawdown', 0.0) * 100 if risk_signal else 0.0) :.1f}%
- VaR(95%): {(risk_signal.get('risk_metrics', {}).get('value_at_risk_95', 0.0) * 100 if risk_signal else 0.0) :.1f}%

##### (2) 风险情况分析
【风险情况分析待补充】

### 三、决策分析汇总 📜
【决策分析汇总待补充】

#### ❗ 重要提示: 预测结果仅供参考，不构成投资建议。股市有风险，投资需谨慎。
    """

    return {
        "report_initial": detailed_analysis
    }




def summary_synthesis_agent(state: AgentState):
    agent_name = "summary_synthesis_agent"
    show_workflow_status("summary_synthesis_agent")

    # -------------------------------------------------------------
    # 1. 获取先前所有环节的日志信息
    # -------------------------------------------------------------
    unique_incoming_messages = {}
    for msg in state["messages"]:
        unique_incoming_messages[msg.name] = msg
    cleaned_messages_for_processing = list(unique_incoming_messages.values())
    show_reasoning_flag = state["metadata"]["show_reasoning"]
    technical_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "technical_analysis_tool")
    fundamentals_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "fundamentals_analysis_tool")
    sentiment_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "online_sentiment_agent")
    valuation_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "valuation_analysis_tool")
    stock_forecast_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "stock_forecast_tool")
    risk_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "risk_assessment_tool")
    tool_based_macro_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "macro_market_agent")
    technical_content = technical_message.content if technical_message else json.dumps(
        {"signal": "error", "details": "Technical message missing"})
    fundamentals_content = fundamentals_message.content if fundamentals_message else json.dumps(
        {"signal": "error", "details": "Fundamentals message missing"})
    sentiment_content = sentiment_message.content if sentiment_message else json.dumps(
        {"signal": "error", "details": "Sentiment message missing"})
    valuation_content = valuation_message.content if valuation_message else json.dumps(
        {"signal": "error", "details": "Valuation message missing"})
    stock_forecast_content = stock_forecast_message.content if stock_forecast_message else json.dumps(
        {"signal": "error", "details": "Stock Forecast message missing"})
    risk_content = risk_message.content if risk_message else json.dumps(
        {"signal": "error", "details": "Risk message missing"})
    tool_based_macro_content = tool_based_macro_message.content if tool_based_macro_message else json.dumps(
        {"signal": "error", "details": "Tool-based Macro message missing"})
    # 宏观新闻分析结果单独获取，用于报告撰写变量输入
    market_wide_news_summary_content = state["data"].get("macro_news_analysis_result", "大盘宏观新闻分析不可用或未提供。")
    # 股票预测结果单独获取，用于报告撰写变量输入
    stock_pred = state["data"].get("prediction_analysis", "股票预测失败")
    stock_pred_result = stock_pred.get("text_report", "股票预测失败")
    # 信息汇总
    agent_signals = {
        "technical_signal": parse_json_signal(technical_content),
        "fundamental_signal": parse_json_signal(fundamentals_content),
        "sentiment_signal": parse_json_signal(sentiment_content),
        "valuation_signal": parse_json_signal(valuation_content),
        "risk_signal": parse_json_signal(risk_content),
        "stock_forecast_signal": parse_json_signal(stock_forecast_content),
        "general_macro_signal": parse_json_signal(tool_based_macro_content),
        "market_wide_news_signal": parse_json_signal(market_wide_news_summary_content)
    }

    # -------------------------------------------------------------
    # 2. 调用 format_decision 来生成初版报告（包含前期分析的各种数据）
    # -------------------------------------------------------------

    formatted_result = format_decision(
        stock_pred_result=stock_pred_result,
        agent_signals=agent_signals,
        market_wide_news_summary=state["data"].get("macro_news_analysis_result", "大盘宏观新闻分析不可用。")
    )

    # -------------------------------------------------------------
    # 3. 大模型调用（总结内容得出结论 + 完善报告）
    # -------------------------------------------------------------

    system_message_content = """
    你是一名负责股票投资建议分析并撰写投资分析报告的经理。
    您的工作目标是汇总并参考各团队的分析结论，指定最终的投资建议（买入/卖出/持有）并说明理由，补充并完善最终的投资分析报告
    
    你的任务主要分为两个部分，请顺序逐步完成：
    
    任务一：综合各团队结果，形成最终的投资建议结论，并给出分析与理由
    在权衡各团队不同信号的方向和时机时，你的思考步骤如下所示：
    1. 基于'基本面分析'、'估值分析'、'技术分析'、'风险评估'、'宏观环境分析中经济环境角度'原始报告数据，各团队分析依据和结果，对这四个模块生成 3-4 句情况分析结论；
    2. 结合步骤1的分析结果，以及原始报告中'宏观环境分析'和'网络情绪分析'的内容，判断最终的投资建议（买入/卖出/持有），各方面的参考权重如下：
        - 基本面分析（20%权重）
        - 估值分析（20%权重）
        - 技术分析（20%权重）
        - 宏观环境分析（15%权重）
        - 网络情绪分析（10%权重）
        - 风险评估（15%权重）
    3. 写一段话说明'决策分析汇总'，其内容应包括以下几个方面
        - 股票维度：分析股票各个维度的核心结论与主要驱动因素，总结公司基本面、估值水平、技术走势、风险暴露中最关键、最影响决策的因素，明确哪些指标或事件是结论的核心依据。
        - 外界因素：分析宏观经济，新闻动态、政策信号与市场交易情绪对结论的边际影响：强调近期宏观经济、新闻事件、监管政策、市场情绪变化与资金流向等对短期股价的反馈程度及其稳定性。
        - 投研总结：分析投资建议形成的最终推导路径，从基本面—估值—行业—情绪—风险的全链路角度，解释为何最终得出“买入 / 卖出 / 持有”的判断，说明长期逻辑和短期信号之间如何做出平衡，强调最关键的驱动与可能的变动因素。
    
    任务二：补充原始报告中的空缺内容，对报告的部分表达进行完善，形成最终版报告
    你的执行步骤如下：
    1. 将 任务一步骤2 生成的最终投资建议（买入/卖出/持有）写入【操作建议待补充】部分，注意写入内容表述为以下3个选项中选一个
        - 买入/增持 💹
        - 卖出/减仓 📉 (若暂未持股，则继续保持不持有状态，不建议买入该股票)
        - 继续持有/场外观望 ➡️
    2. 参考 任务一步骤1 中五个模块的情况分析结论，补充原始报告中【基本面情况分析待补充】、【估值情况分析待补充】、【技术面情况分析待补充】、【风险情况分析待补充】、【经济环境角度关键因素分析待补充】中的内容；
    3. 参考 任务一步骤3 的内容在原始报告中补充【决策分析汇总待补充】部分，分点作答，小标题自拟（注意逻辑通顺，1不宜过长，参考分析方面的要点但不能完全一样），用无序列表 - 符号表示
    4. 优化原始报告中 “股票走势分析”、“4.网络情绪分析”、“5.宏观环境分析” 的语句表达，在不改变核心表述的前提下提升流畅度，其中 “4.网络情绪分析” 采用分点作答，用无序列表 - 符号表示
    注意在补充报告时，请严格遵循以下要求：
    - 报告Markdown结构一定保持不变，原有数据不得更改
    - 仅补充【】内的内容及指定三个分析部分的语句流畅度, 补充后将【】标识删除仅保留内容
    - 其他未提及的报告内容均保持原样
    - 不要将各维度的权重占比情况写进报告里面
    
    在输出的JSON中提供最终报告完整版内容："report_final": "最终报告内容"
    输出示例：{"report_final": "最终报告内容"}
    """

    system_message = {
        "role": "system",
        "content": system_message_content
    }

    user_message_content = f"""
    原始报告是{formatted_result['report_initial']}
    
    各个团队的分析结果与依据 
    - 技术分析团队: 结果是{technical_content}；依据是基于价格、成交量等数据，综合 MACD、RSI、布林带、OBV 等指标及趋势跟踪、均值回归等多种策略生成信号；通过加权组合不同信号，结合多时间框架验证确定结果。
    - 估值分析团队: (1)估值结果是{valuation_content}；依据是采用 DCF 法与所有者收益法测算内在价值，对比当前市值得出估值缺口，判定资产估值信号。(2)股票预测结果是{state["data"].get("predicted_price_data", "未提供股票价格预测结果")}
    - 基本面分析团队: 结果是{fundamentals_content}；依据是从盈利能力、成长能力、财务健康度、估值比率四个维度，通过对比关键指标统计各维度看多/看空信号数量，生成综合判断。
    - 网络情绪分析团队：结果是{sentiment_content}；依据是通过贴吧关键词检索10个相关帖子及帖子下面的评论，基于文本数据进行总结分析。
    - 宏观环境分析团队：(1)经济环境角度是{tool_based_macro_content}; (2)大盘新闻角度是{market_wide_news_summary_content}
    - 风险评估团队：结果是{risk_content}；依据是通过计算波动率、95% 风险价值（VaR）、最大回撤等指标评估市场风险，结合看空/看多/第三方的信号，形成风险分数（0-10 分，越高风险越大）及辩论结果，生成持有、减持、买入或卖出的交易动作建议。
    
    输出一定要是JSON格式
    """

    user_message = {
        "role": "user",
        "content": user_message_content
    }

    llm_interaction_messages = [system_message, user_message]
    llm_response_content = get_chat_completion(llm_interaction_messages)

    current_metadata = state["metadata"]
    current_metadata["current_agent_name"] = agent_name

    if llm_response_content is None:
        llm_response_content = json.dumps({
            "report_final": f"由于大模型分析发生故障，显示原始报告数据内容 \n {formatted_result['report_initial']}"
        })

    if show_reasoning_flag:
        show_agent_reasoning(
            agent_name, f"Final LLM decision JSON: {llm_response_content}")

    try:
        pattern = r'{"report_final":\s*"([^"]+)"'
        match = re.search(pattern, llm_response_content)
        if match:
            final_report_content = match.group(1)
        else:
            final_report_content = f"由于大模型内容解析失败，显示原始报告数据内容{formatted_result['report_initial']}"
        # decision_json = json.loads(llm_response_content) # type: ignore
        # final_report_content = decision_json.get("report_final", f"由于大模型内容解析失败，显示原始报告数据内容{formatted_result['report_initial']}")
        agent_decision_details_value = {"raw_response": llm_response_content}
    except Exception as e:
        logger.error(f"无法解析或处理 portfolio_manager 的 LLM 响应: {e}")
        agent_decision_details_value = {
            "error": f"处理 LLM 决策时出错: {e}",
            "raw_response_snippet": llm_response_content[:200] + "..." # type: ignore
        }
        final_report_content = f"LLM 响应处理失败 (错误: {e})，显示原始报告数据内容 \n {formatted_result['report_initial']}"

    final_decision_message = HumanMessage(
        content=final_report_content,
    )


    final_messages_output = cleaned_messages_for_processing + [final_decision_message]

    serializable_messages = [message_to_dict(m) for m in final_messages_output]
    return_payload = {
        "messages": serializable_messages,
        "data": state["data"],
        "metadata": {
            **state["metadata"],
            f"{agent_name}_decision_details": agent_decision_details_value,
            "agent_reasoning": llm_response_content
        }
    }

    return return_payload


