import os
import sys
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# 상위 디렉토리 모듈 참조
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import (
    get_daily_prices, 
    get_latest_prediction, 
    get_stock_list, 
    get_all_latest_rankings,
    calculate_stock_backtest,
    STOCK_INFO,
    init_db
)

def run_full_pipeline():
    """
    장 마감 후 주식 데이터 크롤링 및 1D-CNN 예측을 순차 실행하는 전체 파이프라인 함수
    """
    print("\n" + "=" * 70)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 전체 주식 파이프라인 실행 시작...")
    print("=" * 70)
    try:
        from crawler.stock_spider import crawl_and_store_all
        from ml_model.train_cnn import train_all_stocks

        # 1. 50개 종목 크롤링 및 DB 적재
        crawl_and_store_all(target_days=120)

        # 2. 50개 종목 1D-CNN 배치 학습 및 예측치 DB 적재
        train_all_stocks(epochs=30)

        print("\n" + "=" * 70)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [v] 전체 주식 파이프라인 자동 갱신 완료!")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"[-] 파이프라인 실행 중 오류 발생: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 수명 주기 관리 및 APScheduler 백그라운드 스케줄러 등록"""
    init_db()

    # 평일(월~금) 한국 시간 16:00 (장 마감 후) 자동 실행 스케줄러 설정
    scheduler = BackgroundScheduler(timezone="Asia/Seoul")
    scheduler.add_job(
        run_full_pipeline,
        CronTrigger(day_of_week="mon-fri", hour=16, minute=0, timezone="Asia/Seoul"),
        id="daily_stock_market_close_pipeline",
        name="평일 16:00 장마감 주식 시세 크롤링 및 1D-CNN 예측 갱신",
        replace_existing=True
    )
    scheduler.start()
    print("[*] APScheduler 가동: 평일(월~금) 16:00 KST 주식 시세 및 AI 예측 자동 갱신 등록 완료.")

    yield

    scheduler.shutdown()
    print("[*] APScheduler 안전하게 종료되었습니다.")

app = FastAPI(
    title="AI 주가 레이더 - Stock AI",
    description="국내 50대 대표 종목 1D-CNN 시계열 주가 예측 및 자동 갱신 플랫폼",
    version="2.1.0",
    lifespan=lifespan
)

# 템플릿 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

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
    (모든 가격 데이터는 정수 반올림 처리)
    """
    stock_code = stock_code.zfill(6)
    if stock_code not in STOCK_INFO:
        raise HTTPException(status_code=404, detail="Stock code not found")

    raw_history = get_daily_prices(stock_code)
    prediction = get_latest_prediction(stock_code)

    if not raw_history:
        return {
            "stock_code": stock_code,
            "stock_name": STOCK_INFO.get(stock_code, stock_code),
            "history": [],
            "prediction": None,
            "message": "데이터가 아직 수집되지 않았습니다. 크롤러를 실행해주세요."
        }

    # 히스토리 가격 정수화
    history = [
        {
            "stock_code": h["stock_code"],
            "date": h["date"],
            "close_price": int(round(float(h["close_price"])))
        }
        for h in raw_history
    ]

    current_price = history[-1]["close_price"]
    base_date = history[-1]["date"]

    # AI 백테스팅 적중률 및 신뢰도 지표 계산
    backtest = calculate_stock_backtest(stock_code, raw_history)

    prediction_data = None
    if prediction:
        pred_1d = int(round(float(prediction["pred_1d"])))
        pred_1w = int(round(float(prediction["pred_1w"])))
        pred_1m = int(round(float(prediction["pred_1m"])))
        pred_base_date = prediction["base_date"]

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
        "prediction": prediction_data,
        "backtest": backtest
    }

@app.get("/api/ranking")
async def api_stock_ranking():
    """
    50대 주요 종목 1D-CNN 예측치 기반 랭킹 큐레이션 데이터를 반환합니다.
    - hero_stock: 오늘의 AI 슈퍼픽 (1주일 예상 상승률 1위)
    - top_1m: 1달 급등 기대주 TOP 5
    - top_1w: 1주일 단기 모멘텀 TOP 5
    - caution_down: 1달 숨고르기/조정 주의 TOP 3
    """
    rankings = get_all_latest_rankings()
    return rankings

@app.post("/api/pipeline/run")
async def trigger_pipeline(background_tasks: BackgroundTasks):
    """
    관리자 수동 갱신용 API: 전체 크롤링 및 1D-CNN 예측 파이프라인을 백그라운드로 즉시 실행합니다.
    """
    background_tasks.add_task(run_full_pipeline)
    return {
        "status": "success",
        "message": "50개 종목 데이터 수집 및 1D-CNN 예측 갱신 파이프라인이 백그라운드에서 시작되었습니다."
    }

if __name__ == "__main__":
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
