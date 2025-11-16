import os
import time
import logging
import backoff
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import SecretStr

# ========== 日志配置 ==========
logger = logging.getLogger(__name__)
SUCCESS_ICON = "✅"
ERROR_ICON = "❌"
WAIT_ICON = "⏳"


# ========== 加载环境变量 ==========
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

BYTEDANCE_API_KEY = os.getenv("BYTEDANCE_API_KEY")
BYTEDANCE_MODEL = os.getenv("BYTEDANCE_MODEL")
BYTEDANCE_BASE_URL = os.getenv("BYTEDANCE_BASE_URL")


if not BYTEDANCE_API_KEY:
    logger.error(f"{ERROR_ICON} 缺少环境变量 BYTEDANCE_API_KEY")
    exit(1)
logger.info(f"{SUCCESS_ICON} 环境变量 BYTEDANCE_API_KEY 加载成功")
# ========== 定义常量 ==========
# ========== 初始化 LLM ==========
llm = ChatOpenAI(
    model = BYTEDANCE_MODEL,
    api_key = SecretStr(BYTEDANCE_API_KEY),
    base_url = BYTEDANCE_BASE_URL,
    timeout=180
)
logger.info(f"{SUCCESS_ICON} ChatOpenAI(Doubao) 模型初始化成功")


# ========== 带重试机制的调用函数 ==========
@backoff.on_exception(
    backoff.expo,                  # 指数退避
    (Exception,),                  # 捕获所有异常
    max_tries=5,                   # 最多重试 5 次
    max_time=300,                  # 最长重试时间 300 秒
)
def generate_content_with_retry(messages):
    """
    调用 LLM 生成内容，带自动重试机制。
    Args:
        messages (list): LangChain 格式的消息列表，例如 [SystemMessage(), HumanMessage()]
    Returns:
        str: LLM 输出文本
    """
    try:
        logger.info(f"{WAIT_ICON} 正在调用 Doubao 模型 API ...")
        logger.debug(f"请求消息: {messages}")

        response = llm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)

        logger.info(f"{SUCCESS_ICON} 调用成功")
        logger.debug(f"响应前 500 字符: {text[:500]}...")
        return text

    except Exception as e:
        err = str(e)
        logger.error(f"{ERROR_ICON} 模型调用失败: {err}")
        raise  # 必须抛出以便 backoff 触发重试


# ========== 通用聊天封装函数 ==========
def get_chat_completion(messages, max_retries=3, initial_retry_delay=1):
    """
    通用聊天接口，封装消息结构 + 重试逻辑。
    Args:
        messages (list[dict] or list[Message]):
            支持 OpenAI 风格字典或 LangChain Message 对象。
        max_retries (int): 最大重试次数
        initial_retry_delay (int): 初始重试延迟秒数
    Returns:
        str: 模型回复文本
    """
    # 支持 OpenAI 格式自动转换
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            role, content = msg.get("role"), msg.get("content")
            if role == "system":
                formatted_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                formatted_messages.append(AIMessage(content=content))
            else:
                formatted_messages.append(HumanMessage(content=content))
        else:
            formatted_messages.append(msg)

    retries = 0
    while retries < max_retries:
        try:
            return generate_content_with_retry(formatted_messages)
        except Exception as e:
            retries += 1
            delay = initial_retry_delay * (2 ** (retries - 1))
            logger.warning(f"{ERROR_ICON} 第 {retries} 次调用失败: {e}，{delay}s 后重试...")
            time.sleep(delay)
    logger.error(f"{ERROR_ICON} 达到最大重试次数 {max_retries}，任务失败。")
    return None


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "你是一个专业的投研分析助手。"},
        {"role": "user", "content": "请简要分析A股市场的短期趋势。"}
    ]

    answer = get_chat_completion(messages)
    print("\n=== 模型回复 ===\n", answer)
