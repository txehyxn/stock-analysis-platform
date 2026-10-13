import os
import sys
from typing import Dict, Any, Optional

# 상위 디렉토리 참조
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from predict import predict_stock, run_predict_pipeline
from database.db_manager import STOCK_INFO

def predict_stock_rf(
    stock_code: str,
    df: Optional[Any] = None,
    save_csv: bool = True
) -> Optional[Dict[str, Any]]:
    """
    공식 주 예측 엔진(predict.py Huber 손실 직접 타깃 모델)을 호출합니다.
    """
    return predict_stock(stock_code, df=df, save_csv=save_csv)


def predict_all_stocks_rf() -> Dict[str, Dict[str, Any]]:
    """
    50개 전체 지원 종목에 대해 Huber 직접 타깃 예측 파이프라인을 실행합니다.
    """
    return run_predict_pipeline()


if __name__ == "__main__":
    predict_all_stocks_rf()
