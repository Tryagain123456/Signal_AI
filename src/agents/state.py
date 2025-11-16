from typing import Annotated, Any, Dict, Sequence, TypedDict

import operator
from langchain_core.messages import BaseMessage
import json
from src.utils.logging_config import setup_logger

# 设置日志记录，日志初始化（创建名为 agent_state 的日志对象，后续所有状态、结果都会通过它打印）
logger = setup_logger('agent_state')

def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {**a, **b}


# 智能体的状态定义
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add] # 存储智能体间交互的信息的列表（新消息追加到末尾）
    data: Annotated[Dict[str, Any], merge_dicts] # 存储业务核心数据
    metadata: Annotated[Dict[str, Any], merge_dicts] # 存储运行配置信息


# 工作流中各个 agent 的工作状态
def show_workflow_status(agent_name: str, status: str = "processing"):
    if status == "processing":
        logger.info(f"🔄 {agent_name} 正在分析中")
    else:
        logger.info(f"✅ {agent_name} 分析完成")

# agent的推理结果日志
def show_agent_reasoning(output, agent_name):
    def convert_to_serializable(obj):
        # 处理各种复杂数据类型，将其转为 JSON 能识别的格式（如字典、列表）
        if hasattr(obj, 'to_dict'):   # 处理 Pandas 的 Series/DataFrame
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):  # 处理自定义对象
            return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)):  # 基础类型直接返回
            return obj
        elif isinstance(obj, (list, tuple)): # 列表/元组递归处理每个元素
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):   # 字典递归处理每个键值对
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)  # 其他类型转为字符串

    # 各个智能体的推理结果
    logger.info(f"{'='*20} {agent_name} 分析结果 {'='*20}")
    if isinstance(output, (dict, list)):
        # 若结果是字典或列表，优先按 JSON 格式化
        serializable_output = convert_to_serializable(output)
        logger.info(json.dumps(serializable_output, indent=2))
    else:
        # 若结果是字符串：解析为 JSON or 直接打印原始字符串（如纯文本分析结论）
        try:
            parsed_output = json.loads(output)
            logger.info(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            # Fallback to original string if not valid JSON
            logger.info(output)
