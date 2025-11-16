
from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.logging_config import setup_logger
from src.Crawl_Comments.crawl_function import all_crawl_process
from src.tools.openrouter_config import get_chat_completion

import json
from datetime import datetime, timedelta

# 设置日志记录
logger = setup_logger('online_sentiment_agent')


# 爬虫函数
def crawl_agent(data):
    keywords = [data.get("stock_name")]
    if keywords is None or (isinstance(keywords, str) and not keywords.strip()):
        error_msg = "❌ 输入数据缺少必要字段 'keywords'"
        logger.error(error_msg)
        crawl_result = None
        return crawl_result
    try:
        crawl_result = all_crawl_process("tieba", keywords)
        if not crawl_result:
            logger.warning("❌ 爬取百度贴吧结果为空")
            crawl_result = None
            return crawl_result
        logger.info("✅ 爬取百度贴吧评论成功")
        return crawl_result
    except Exception as e:
        print(f"❌ 爬取百度贴吧评论失败，失败原因：{e}")
        crawl_result = None
        return crawl_result

def sentiment_analyze(crawl_result):
    system_message = {
        "role": "system",
        "content": """ 你是一位专业的 A 股网络情绪分析师，擅长从网络内容（如贴吧、论坛言论等）中提取市场情绪，精准评估其对目标股票的潜在影响。
        请基于提供的网络内容，按以下要求完成分析：
        
        你的分析结果应该包括：
        1. 股票网络情绪评估（结果用中文即可）：明确标注为积极 (positive)、中性 (neutral) 或消极 (negative)
        2. 情绪对目标股票的影响（结果用中文即可）：明确标注为利好 (positive)、中性 (neutral) 或利空 (negative)
        3. 关键影响因素：列出 3-5 个最重要的情绪相关因素（聚焦网络舆论核心话题）
        4. 详细推理：解释这些情绪因素为何会影响目标股票，逻辑需贴合 A 股市场特点
        
        分析时重点关注的网络情绪维度：
        1. 言论倾向：正面 / 负面 / 中性言论的占比及强度
        2. 核心话题：用户集中讨论的股票相关关键事件（如业绩预期、政策关联、风险传闻等）
        3. 舆论集中度：单一情绪（积极 / 消极）的传播范围和讨论热度
        4. 信息可信度：网络言论的事实依据、传播者身份可信度
        5. 市场关联度：情绪话题与股票基本面、短期走势的关联性
        
        请确保你的分析：
        1. 完全基于提供的网络内容，不添加外部未提及信息
        2. 结合目标股票的行业属性和公司特点，避免泛泛而谈
        3. 情绪评估与影响判断逻辑一致，推理过程可追溯
        4. 关键影响因素聚焦 “情绪相关”，不偏离网络舆论核心 
        5. 呈现分析结论时，不需要具体列举观点是有哪个帖子xx或评论xx推导的
        6. 具体的推到过程分点作答，分点可以用markdown无序列表的形式呈现，但不要有加粗、标题等形式
        """
    }

    user_message = {
        "role": "user",
        "content": f"请分析以下贴吧帖子和评论内容中的市场情绪，并评估当前市场情绪对相关A股上市公司的影响：\n\n{crawl_result}\n\n请以JSON格式返回结果，包含以下字段：sentiment_signal（股票网络情绪评估：积极/中性/消极）、sentiment_impact（情绪对目标股票的影响：积极/中性/消极）、key_factors（关键因素数组）、reasoning（详细推理）。"
    }

    try:
        # 获取LLM分析结果
        result = get_chat_completion([system_message, user_message])
        if result is None:
            logger.error("LLM分析失败，无法获取宏观分析结果")
            return {
                "sentiment_signal": "neutral",
                "sentiment_impact": "neutral",
                "key_factors": [],
                "reasoning": "LLM分析失败，无法获取情感分析结果，故为中立态度"
            }

        # 解析JSON结果
        try:
            # 尝试直接解析
            analysis_result = json.loads(result.strip())
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试提取JSON部分
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if json_match:
                try:
                    analysis_result = json.loads(json_match.group(1).strip())
                except:
                    logger.error("无法解析情绪分析中的JSON结果")
                    return {
                        "sentiment_signal": "neutral",
                        "sentiment_impact": "neutral",
                        "key_factors": [],
                        "reasoning": "无法解析LLM返回的JSON结果，获取情感分析结果失败"
                    }
            else:
                # 如果没有找到JSON，返回默认结果
                logger.error("情绪分析中LLM未返回有效的JSON格式结果")
                return {
                    "sentiment_signal": "neutral",
                    "sentiment_impact": "neutral",
                    "key_factors": [],
                    "reasoning": "LLM未返回有效的JSON格式结果，获取情感分析结果失败"
                }
        return analysis_result

    except Exception as e:
        logger.error(f"宏观分析出错: {e}")
        return {
            "macro_environment": "neutral",
            "impact_on_stock": "neutral",
            "key_factors": [],
            "reasoning": f"分析过程中出错: {str(e)}"
        }

def online_sentiment_agent(state: AgentState):
    show_workflow_status("online_sentiment_agent")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    crawl_result = crawl_agent(data)
    # 如果爬取失败
    if not crawl_result:
        message_content = {
            "sentiment_signal": "neutral",
            "sentiment_impact": "neutral",
            "key_factors": [],
            "reasoning": "未获取百度贴吧评论数据，无法进行情感分析，故为中立态度"
        }
    else:
        # 获取情绪分析结果
        macro_analysis = sentiment_analyze(crawl_result)
        message_content = macro_analysis

    # 如果需要显示推理过程
    if show_reasoning:
        show_agent_reasoning(message_content, "Sentiment Analysis Agent")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    # 创建消息
    message = HumanMessage(
        content=json.dumps(message_content),
        name="online_sentiment_agent",
    )

    show_workflow_status("online_sentiment_agent", "completed")

    return {
        "messages": state["messages"] + [message],
        "data": {
            **data,
            "sentiment_analysis": message_content
        },
        "metadata": state["metadata"],
    }


if __name__ == "__main__":
    from typing import Sequence, Dict, Any
    from langchain_core.messages import BaseMessage, HumanMessage
    # 测试用
    # 1. 初始化messages（空列表或包含初始消息）
    initial_messages: Sequence[BaseMessage] = [
        HumanMessage(content="我想看看万向钱潮是否值得投资")  # 示例初始消息
    ]

    # 2. 初始化data（包含业务核心数据，如ticker）
    initial_data: Dict[str, Any] = {
        "ticker": "000559",
        "stock_name": "万向钱潮股票"
    }

    # 3. 初始化metadata（运行配置，如爬虫参数、超时设置等）
    initial_metadata: Dict[str, Any] = {
        "show_reasoning": True
    }

    # 4. 组合为AgentState实例（TypedDict本质是字典，直接按键初始化）
    test_state: AgentState = {
        "messages": initial_messages,
        "data": initial_data,
        "metadata": initial_metadata
    }

    try:
        result = online_sentiment_agent(test_state)
        print("测试结果：", result)
    except KeyError as e:
        print(f"缺少必要字段：{e}")
    except Exception as e:
        print(f"测试出错：{e}")