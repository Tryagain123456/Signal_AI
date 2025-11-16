# kronos_api.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import torch
import logging
import os
from dotenv import load_dotenv

from model import Kronos, KronosTokenizer, KronosPredictor


# ================================================================
# 2. 日志配置
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="logs/kronos_api.log",  # 保存路径
    filemode="a"  # a=追加, w=覆盖
)
logger = logging.getLogger("kronos_api")

# ================================================================
# 3. 定义请求/响应模型
# ================================================================
class KlineData(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: Optional[float] = None

class PredictRequest(BaseModel):
    kline_data: List[KlineData] = Field(..., description="历史 K 线数据，建议不少于 10 条")
    pred_len: int = Field(default=10, ge=1, le=200, description="预测步长（未来预测长度）")
    freq: str = Field(default="1h", description="时间间隔，如 '1h', '5min', '1d'")
    T: float = Field(default=1.0, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.5, le=1.0)
    sample_count: int = Field(default=1, ge=1, le=5)

class PredictResponse(BaseModel):
    code: int
    message: str
    data: Dict[str, Dict[str, float]]

# ================================================================
# 4. 模型加载（全局单例）
# ================================================================

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
env_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=env_path)
MODEL_PATH = os.environ.get("KRONOS_MODEL_PATH")
TOKENIZER_PATH = os.environ.get("KRONOS_TOKENIZER_PATH")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = None
tokenizer = None
predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, predictor
    try:
        logger.info("🚀 启动服务，加载模型中...")
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PATH)
        model = Kronos.from_pretrained(MODEL_PATH).to(device)
        predictor = KronosPredictor(model, tokenizer, device=device)
        logger.info(f"✅ 模型加载完成，使用设备: {device}")
        yield  # 这里进入应用正常运行阶段
    finally:
        logger.info("🛑 服务关闭，释放资源...")
        # 可以在此释放 GPU 内存
        del predictor, model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("✅ 资源释放完成")



app = FastAPI(
    title="Kronos 量化预测 API",
    description="基于 Kronos 模型的 K 线时间序列预测服务。",
    version="1.1.0",
    lifespan=lifespan
)


# ================================================================
# 5. 工具函数
# ================================================================
def parse_timedelta(freq: str) -> pd.Timedelta:
    """安全解析时间间隔字符串，如 1h, 5min, 1d。"""
    try:
        return pd.Timedelta(freq)
    except Exception:
        raise HTTPException(status_code=400, detail=f"无效时间频率参数：{freq}")


def get_predictor() -> KronosPredictor:
    """依赖注入，确保 predictor 已加载。"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="模型未加载完成")
    return predictor


# ================================================================
# 6. 主预测接口
# ================================================================
@app.post("/api/kronos/predict", response_model=PredictResponse, summary="K 线预测")
async def predict_kline(request: PredictRequest, predictor: KronosPredictor = Depends(get_predictor)):
    try:
        if len(request.kline_data) < 10:
            raise HTTPException(status_code=400, detail="历史数据不足（至少需要 10 条）")

        df = pd.DataFrame([x.dict() for x in request.kline_data])
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"缺少列：{col}")

        # 生成历史时间戳
        delta = parse_timedelta(request.freq)
        end_time = pd.Timestamp.now()
        start_time = end_time - delta * (len(df) - 1)
        df["timestamps"] = pd.date_range(start=start_time, periods=len(df), freq=request.freq)

        # 生成未来预测时间戳 (类型为 DatetimeIndex)
        y_timestamp_index = pd.date_range(
            start=end_time + delta,
            periods=request.pred_len,
            freq=request.freq
        )

        # ================================================================
        
        # 1. 准备模型输入的 df：
        #    根据 Jupyter 测试，df 必须包含时间戳列
        #    但根据错误 1，df 必须排除含 NaN 的 'amount' 列
        model_input_cols = required_cols + ["timestamps"]
        df_model_input = df[model_input_cols]

        # 2. 准备 x_timestamp (类型必须是 Series)
        x_timestamps_series = pd.Series(df["timestamps"])

        # 3. 准备 y_timestamp (类型必须是 Series，修复 'dt' 错误)
        y_timestamps_series = pd.Series(y_timestamp_index)
        # ================================================================


        # 模型预测
        with torch.inference_mode():
            pred_df = predictor.predict(
                df=df_model_input,                 # <--- 1. 使用筛选后的 DF
                x_timestamp=x_timestamps_series,   # <--- 2. 使用 x_timestamp Series
                y_timestamp=y_timestamps_series,   # <--- 3. 使用 y_timestamp Series
                pred_len=request.pred_len,
                T=request.T,
                top_p=request.top_p,               # <--- 确保这里是 top_p
                sample_count=request.sample_count
            )

        # 格式化输出
        result = {
            str(ts): {
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": round(float(row["volume"]), 2),
            }
            for ts, row in pred_df.iterrows()
        }

        return PredictResponse(code=200, message="预测成功", data=result)

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("预测失败：%s", str(e))
        return PredictResponse(code=500, message=f"预测失败: {str(e)}", data={})


# ================================================================
# 7. 健康检查
# ================================================================
@app.get("/api/health", summary="服务健康检查")
async def health_check():
    return {
        "status": "healthy" if predictor else "uninitialized",
        "device": device,
        "model_loaded": predictor is not None,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ================================================================
# 8. 本地启动（仅调试）
# ================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app="kronos_api:app",
        host="0.0.0.0",
        port=7661,
        reload=False,
        workers=1
    )
