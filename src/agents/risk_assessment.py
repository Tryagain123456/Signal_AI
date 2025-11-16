import math

from langchain_core.messages import HumanMessage

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.api import prices_to_df

import json
import ast



def risk_assessment_tool(state: AgentState):
    """Responsible for risk management"""
    show_workflow_status("risk_assessment_tool")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]

    prices_df = prices_to_df(data["prices"])

    debate_message = next(
        msg for msg in state["messages"] if msg.name == "tripartite_judgment_agent")

    try:
        debate_results = json.loads(debate_message.content)
    except Exception as e:
        debate_results = ast.literal_eval(debate_message.content)

    returns = prices_df['close'].pct_change().dropna()
    daily_vol = returns.std()
    volatility = daily_vol * (252 ** 0.5)

    # 计算波动率的历史分布
    rolling_std = returns.rolling(window=120).std() * (252 ** 0.5)
    volatility_mean = rolling_std.mean()
    volatility_std = rolling_std.std()
    volatility_percentile = (volatility - volatility_mean) / volatility_std

    var_95 = returns.quantile(0.05)
    # 使用60天窗口计算最大回撤
    max_drawdown = (
        prices_df['close'] / prices_df['close'].rolling(window=60).max() - 1).min()

    market_risk_score = 0

    if volatility_percentile > 1.5:     # 高于1.5个标准差
        market_risk_score += 2
    elif volatility_percentile > 1.0:   # 高于1个标准差
        market_risk_score += 1

    if var_95 < -0.03:
        market_risk_score += 2
    elif var_95 < -0.02:
        market_risk_score += 1

    if max_drawdown < -0.20:  # Severe drawdown
        market_risk_score += 2
    elif max_drawdown < -0.10:
        market_risk_score += 1

    bull_confidence = debate_results["bull_confidence"]
    bear_confidence = debate_results["bear_confidence"]
    debate_confidence = debate_results["confidence"]

    confidence_diff = abs(bull_confidence - bear_confidence)
    if confidence_diff < 0.1:
        market_risk_score += 1
    if debate_confidence < 0.3:
        market_risk_score += 1

    risk_score = min(round(market_risk_score), 10)

    # 6. 根据风险分数和辩论信号生成交易操作
    debate_signal = debate_results["signal"]

    if risk_score >= 9:
        trading_action = "hold"
    elif risk_score >= 7:
        trading_action = "reduce"
    else:
        if debate_signal == "bullish" and debate_confidence > 0.5:
            trading_action = "buy"
        elif debate_signal == "bearish" and debate_confidence > 0.5:
            trading_action = "sell"
        else:
            trading_action = "hold"

    message_content = {
        "risk_score": risk_score,
        "trading_action": trading_action,
        "risk_metrics": {
            "volatility": float(volatility),
            "value_at_risk_95": float(var_95),
            "max_drawdown": float(max_drawdown),
            "market_risk_score": market_risk_score
        },
        "debate_analysis": {
            "bull_confidence": bull_confidence,
            "bear_confidence": bear_confidence,
            "debate_confidence": debate_confidence,
            "debate_signal": debate_signal
        },
        "reasoning": f"Risk Score {risk_score}/10: Market Risk={market_risk_score}, "
                     f"Volatility={volatility:.2%}, VaR={var_95:.2%}, "
                     f"Max Drawdown={max_drawdown:.2%}, Debate Signal={debate_signal}"
    }

    message = HumanMessage(
        content=json.dumps(message_content),
        name="risk_assessment_tool",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "Risk Management Agent")
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("risk_assessment_tool", "completed")
    return {
        "messages": state["messages"] + [message],
        "data": {
            **data,
            "risk_analysis": message_content
        },
        "metadata": state["metadata"],
    }
