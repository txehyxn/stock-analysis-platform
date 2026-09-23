import os
import sys
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import jwt
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
    create_user,
    authenticate_user,
    get_user_by_id,
    toggle_user_favorite,
    get_user_favorites,
    is_user_favorite,
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

# ====================================================
# JWT 인증 설정 및 헬퍼 함수
# ====================================================
JWT_SECRET = os.getenv("JWT_SECRET", "toss-stock-ai-secret-key-2026-very-secure")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 7

class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

def create_access_token(user_id: int, email: str, username: str) -> str:
    """사용자 정보로 7일간 유효한 JWT 토큰을 발급합니다."""
    payload = {
        "user_id": user_id,
        "email": email,
        "username": username,
        "exp": datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def get_current_user_optional(request: Request) -> Optional[dict]:
    """Authorization 헤더(Bearer 토큰)를 파싱하여 유효한 사용자 정보를 반환합니다."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user = get_user_by_id(payload["user_id"])
        return user
    except Exception:
        return None

# ====================================================
# 회원가입 및 로그인 인증 엔드포인트
# ====================================================
@app.post("/api/auth/register")
async def api_register(req: RegisterRequest):
    """신규 회원가입을 처리하고 JWT 토큰을 발급합니다."""
    if not req.email or not req.username or not req.password:
        raise HTTPException(status_code=400, detail="모든 항목을 입력해주세요.")
    if len(req.password) < 4:
        raise HTTPException(status_code=400, detail="비밀번호는 최소 4자 이상이어야 합니다.")
    
    user = create_user(req.email, req.username, req.password)
    if not user:
        raise HTTPException(status_code=400, detail="이미 등록된 이메일이거나 회원가입에 실패했습니다.")

    token = create_access_token(user["id"], user["email"], user["username"])
    return {
        "status": "success",
        "message": "회원가입이 완료되었습니다.",
        "token": token,
        "user": user
    }

@app.post("/api/auth/login")
async def api_login(req: LoginRequest):
    """로그인을 검증하고 JWT 토큰을 발급합니다."""
    user = authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 일치하지 않습니다.")

    token = create_access_token(user["id"], user["email"], user["username"])
    return {
        "status": "success",
        "message": "로그인에 성공했습니다.",
        "token": token,
        "user": user
    }

@app.get("/api/auth/me")
async def api_auth_me(request: Request):
    """현재 로그인된 사용자 정보를 반환합니다."""
    user = get_current_user_optional(request)
    if not user:
        raise HTTPException(status_code=401, detail="로그인이 필요하거나 세션이 만료되었습니다.")
    return {"user": user}

# ====================================================
# 관심 종목(즐겨찾기) 엔드포인트
# ====================================================
@app.get("/api/favorites")
async def api_get_favorites(request: Request):
    """
    로그인한 사용자가 찜한 관심 종목 종합 리스트를 반환합니다.
    (실시간 종가, 1W/1M AI 목표가, 백테스팅 적중률 포함)
    """
    user = get_current_user_optional(request)
    if not user:
        raise HTTPException(status_code=401, detail="로그인이 필요한 기능입니다.")

    fav_codes = get_user_favorites(user["id"])
    favorites_list = []
    
    for code in fav_codes:
        name = STOCK_INFO.get(code, code)
        raw_hist = get_daily_prices(code)
        pred = get_latest_prediction(code)
        
        if raw_hist:
            cur_price = int(round(float(raw_hist[-1]["close_price"])))
            base_date = raw_hist[-1]["date"]
            bt = calculate_stock_backtest(code, raw_hist)
        else:
            cur_price = 0
            base_date = "-"
            bt = {"hit_ratio": 0, "grade": "분석중", "status_color": "slate"}
        
        pred_1w = int(round(float(pred["pred_1w"]))) if pred else cur_price
        pred_1m = int(round(float(pred["pred_1m"]))) if pred else cur_price
        chg_1w = round(((pred_1w - cur_price) / cur_price) * 100, 1) if cur_price > 0 else 0.0
        chg_1m = round(((pred_1m - cur_price) / cur_price) * 100, 1) if cur_price > 0 else 0.0

        favorites_list.append({
            "code": code,
            "name": name,
            "current_price": cur_price,
            "base_date": base_date,
            "target_1w": pred_1w,
            "change_1w_pct": chg_1w,
            "target_1m": pred_1m,
            "change_1m_pct": chg_1m,
            "hit_ratio": bt.get("hit_ratio", 0),
            "grade": bt.get("grade", "보통"),
            "status_color": bt.get("status_color", "blue")
        })

    return {
        "favorites": favorites_list,
        "count": len(favorites_list)
    }

@app.post("/api/favorites/{stock_code}/toggle")
async def api_toggle_favorite(stock_code: str, request: Request):
    """특정 종목을 사용자의 관심 종목에 등록하거나 해제(토글)합니다."""
    user = get_current_user_optional(request)
    if not user:
        raise HTTPException(status_code=401, detail="로그인이 필요한 기능입니다.")
    
    stock_code = stock_code.zfill(6)
    if stock_code not in STOCK_INFO:
        raise HTTPException(status_code=404, detail="존재하지 않는 종목 코드입니다.")

    is_fav = toggle_user_favorite(user["id"], stock_code)
    return {
        "status": "success",
        "stock_code": stock_code,
        "is_favorite": is_fav,
        "message": "관심 종목에 추가되었습니다." if is_fav else "관심 종목에서 제거되었습니다."
    }

# ====================================================
# 주식 상세 및 랭킹 조회 엔드포인트
# ====================================================
@app.get("/api/stock/{stock_code}")
async def api_stock_detail(stock_code: str, request: Request):
    """
    특정 종목의 과거 종가 이력 및 1D-CNN 예측 데이터(1일/1주/1달)를 반환합니다.
    (모든 가격 데이터는 정수 반올림 처리, 로그인 유저의 찜 여부 포함)
    """
    stock_code = stock_code.zfill(6)
    if stock_code not in STOCK_INFO:
        raise HTTPException(status_code=404, detail="Stock code not found")

    raw_history = get_daily_prices(stock_code)
    prediction = get_latest_prediction(stock_code)

    user = get_current_user_optional(request)
    is_fav = is_user_favorite(user["id"], stock_code) if user else False

    if not raw_history:
        return {
            "stock_code": stock_code,
            "stock_name": STOCK_INFO.get(stock_code, stock_code),
            "history": [],
            "prediction": None,
            "is_favorite": is_fav,
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
        "backtest": backtest,
        "is_favorite": is_fav
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
