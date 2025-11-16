from typing import Any, Dict, Sequence, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status

from src.utils.logging_config import setup_logger
import pandas as pd
import os

logger = setup_logger('stock_forecast_tool')

from datetime import datetime, timedelta

try:
    from Kronos.kronos_predictor import kronos_predict, analyze_prediction_df
except ImportError as e:
    print(f"预测模型 导入失败: {e}")
    print("请确保 Gen_Portfolio 的父目录在 PYTHONPATH 中")
    # 优雅地处理导入失败，避免 agent 崩溃
    kronos_predict = None
    analyze_prediction_df = None



# ---------------------------------------------------------------------
# 智能体函数定义
# ---------------------------------------------------------------------
def stock_forecast_tool(state: AgentState) -> AgentState:
    """
    LangGraph 智能体：执行股票预测任务，更新 AgentState。
    """
    if kronos_predict is None or analyze_prediction_df is None:
        msg = "❌ 关键错误: 预测模块(Kronos)未能成功导入。请检查 PYTHONPATH 和依赖。"
        logger.critical(msg)
        state["messages"].append(AIMessage(content=msg)) # type: ignore
        return state

    # 从 state.metadata 或 state.data 中读取参数（允许外部控制）
    symbol = state["data"].get("ticker", "601519")
    # 如果 metadata 没有指定日期，则自动设置默认范围
    today = datetime.today()

    # 默认起始时间：过去三年
    start_date = (today - timedelta(days=3*365)).strftime("%Y-%m-%d")
    # 默认结束时间：今天
    end_date = today.strftime("%Y-%m-%d")

    # 默认预测区间：未来三个月（约90天）
    pred_len = 90
    fig = None
    hist_df = None
    pred_df = None
    result = None
    try:
        hist_df, pred_df, fig = kronos_predict(
            symbol=symbol,
            start_date_str=start_date,
            end_date_str=end_date,
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=3
        )
        # 分析预测结果
        if pred_df is None or pred_df.empty:
            msg = f"⚠️ 股票 {symbol} 预测失败或无数据。"
            logger.warning(msg)
            state["messages"].append(AIMessage(content=msg)) # type: ignore
            return state

        # 只有拿到有效预测结果才做进一步分析
        try:
            result = analyze_prediction_df(pred_df, freq_per_year=252, bootstrap_iters=2000)
            logger.debug("analyze_prediction_df 完成：%s", result)
        except Exception as e:
            logger.warning("analyze_prediction_df 报错：%s，跳过量化分析", e)
            result = None
        # print(result)


        # 构造结果汇总信息
        summary_text = (
            f"✅ 股票 {symbol} 预测完成。\n"
            f"预测周期：{start_date} 至 {end_date}\n"
            f"预测未来 {pred_len} 天价格趋势分析已生成。\n"
        )
        fig_dict = fig.to_dict() if fig else None
        # 更新 state
        state["data"].update({
            "prediction_analysis": result,
            "prediction_plotly_fig": fig_dict # <-- 存储字典而不是对象
        })

        state["messages"].append(AIMessage(content=summary_text)) # type: ignore

        show_workflow_status("股票预测已完成 ✅")
        return state

    except Exception as e:
        error_msg = f"❌ 股票预测过程中出现错误: {e}"
        logger.exception(error_msg)
        state["messages"].append(AIMessage(content=error_msg)) # type: ignore
        return state

