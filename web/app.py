import os
import sys
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

# 상위 디렉토리 모듈 참조
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import (
    get_daily_prices, 
    get_latest_prediction, 
    get_stock_list, 
    STOCK_INFO,
    init_db
)

app = FastAPI(
    title="Stock Analysis & 1D-CNN Prediction Platform",
    description="실시간 네이버 증권 데이터 및 PyTorch 1D-CNN 다중 회귀 예측 대시보드",
    version="1.0.0"
)

# 템플릿 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """메인 대시보드 뷰 렌더링"""
    stocks = get_stock_list()
    default_stock = stocks[0]["code"] if stocks else "005930"
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "stocks": stocks,
            "default_stock": default_stock,
            "default_stock_name": STOCK_INFO.get(default_stock, default_stock)
        }
    )

@app.get("/api/stocks")
async def api_stock_list():
    """지원 종목 리스트 반환"""
    return get_stock_list()

def add_business_days(start_date_str: str, n_days: int) -> str:
    """영업일(주말 제외) 기준 N일 뒤의 예상 날짜 문자열(YYYY-MM-DD)을 계산합니다."""
    try:
        dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    except ValueError:
        return start_date_str
    
    added = 0
    current = dt
    while added < n_days:
        current += timedelta(days=1)
        if current.weekday() < 5:  # 월(0) ~ 금(4)
            added += 1
    return current.strftime("%Y-%m-%d")

@app.get("/api/stock/{stock_code}")
async def api_stock_detail(stock_code: str):
    """
    특정 종목의 과거 종가 이력 및 1D-CNN 예측 데이터(1일/1주/1달)를 반환합니다.
    """
    stock_code = stock_code.zfill(6)
    if stock_code not in STOCK_INFO:
        raise HTTPException(status_code=404, detail="Stock code not found")

    history = get_daily_prices(stock_code)
    prediction = get_latest_prediction(stock_code)

    if not history:
        return {
            "stock_code": stock_code,
            "stock_name": STOCK_INFO.get(stock_code, stock_code),
            "history": [],
            "prediction": None,
            "message": "데이터가 아직 수집되지 않았습니다. 크롤러를 실행해주세요."
        }

    current_price = history[-1]["close_price"]
    base_date = history[-1]["date"]

    prediction_data = None
    if prediction:
        pred_1d = prediction["pred_1d"]
        pred_1w = prediction["pred_1w"]
        pred_1m = prediction["pred_1m"]
        pred_base_date = prediction["base_date"]

        # 예측 시점 날짜 산출 (1영업일 뒤, 5영업일 뒤(1주), 20영업일 뒤(1달))
        date_1d = add_business_days(pred_base_date, 1)
        date_1w = add_business_days(pred_base_date, 5)
        date_1m = add_business_days(pred_base_date, 20)

        prediction_data = {
            "id": prediction["id"],
            "base_date": pred_base_date,
            "base_price": current_price,
            "pred_1d": pred_1d,
            "pred_1w": pred_1w,
            "pred_1m": pred_1m,
            "change_1d_pct": round(((pred_1d - current_price) / current_price) * 100, 2),
            "change_1w_pct": round(((pred_1w - current_price) / current_price) * 100, 2),
            "change_1m_pct": round(((pred_1m - current_price) / current_price) * 100, 2),
            "date_1d": date_1d,
            "date_1w": date_1w,
            "date_1m": date_1m,
            "created_at": str(prediction.get("created_at", ""))
        }

    return {
        "stock_code": stock_code,
        "stock_name": STOCK_INFO.get(stock_code, stock_code),
        "current_price": current_price,
        "base_date": base_date,
        "history": history,
        "prediction": prediction_data
    }

if __name__ == "__main__":
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
