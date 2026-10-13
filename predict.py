import os
import sys
import glob
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler

# Windows 콘솔 UTF-8 출력 보장
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 프로젝트 루트 및 모듈 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from database.db_manager import get_daily_prices, save_prediction, STOCK_INFO

# ==========================================
# 폴더 경로 설정 (data: 원본, predict: 예측 결과)
# ==========================================
data_folder = os.path.join(BASE_DIR, "data")
predict_folder = os.path.join(BASE_DIR, "predict")

os.makedirs(data_folder, exist_ok=True)
os.makedirs(predict_folder, exist_ok=True)

# 피처 목록 정의 (누수 없는 t 및 t-1 시점 기준 표준 피처)
FEATURE_COLS = [
    "disp_ma5",
    "disp_ma20",
    "disp_ma60",
    "vol_10d",
    "ret_lag1",
    "ret_lag5",
    "candle_body",
    "candle_shadow"
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    미래 데이터 누수(Data Leakage)가 전혀 없는 순수 과거(t, t-1) 기준 피처를 생성합니다.
    - 이동평균 수렴확산(MA5/20/60 이격도)
    - 역사적 변동성(10일 로그 수익률 std)
    - 과거 모멘텀(Lag 1, 5일 수익률)
    - 캔들 몸통/꼬리 비율(전일 t-1 기준)
    """
    df = df.copy()
    close = df["Close"].astype(float)

    # 1. 이동평균 수렴확산 (MA5/20/60 이격도)
    ma5 = close.rolling(window=5, min_periods=2).mean()
    ma20 = close.rolling(window=20, min_periods=5).mean()
    ma60 = close.rolling(window=60, min_periods=10).mean()

    df["disp_ma5"] = ((close - ma5) / ma5).fillna(0.0)
    df["disp_ma20"] = ((close - ma20) / ma20).fillna(0.0)
    df["disp_ma60"] = ((close - ma60) / ma60).fillna(0.0)

    # 2. 일별 로그수익률 및 역사적 변동성 (10일 로그 수익률 rolling std)
    log_ret = np.log(close / close.shift(1))
    df["vol_10d"] = log_ret.rolling(window=10, min_periods=3).std().fillna(0.0)

    # 3. 과거 모멘텀 (Lag 1, 5일 수익률)
    df["ret_lag1"] = log_ret.fillna(0.0)
    df["ret_lag5"] = (np.log(close / close.shift(5))).fillna(0.0)

    # 4. 캔들 몸통/꼬리 비율 (전일 t-1 기준 시점 정렬로 미래 누수 방지)
    if all(col in df.columns for col in ["Open", "High", "Low"]):
        open_p = df["Open"].astype(float)
        high_p = df["High"].astype(float)
        low_p = df["Low"].astype(float)
        body = ((close - open_p) / open_p).shift(1).fillna(0.0)
        shadow = ((high_p - np.maximum(open_p, close)) / close).shift(1).fillna(0.0)
        df["candle_body"] = body
        df["candle_shadow"] = shadow
    else:
        # OHLC 컬럼 부재 시(종가 시세) 전일 등락률 및 가격 스윙 대체
        df["candle_body"] = log_ret.shift(1).fillna(0.0)
        df["candle_shadow"] = (np.abs(log_ret)).shift(1).fillna(0.0)

    return df


def predict_stock(
    stock_code: str,
    df: Optional[pd.DataFrame] = None,
    save_csv: bool = True
) -> Optional[Dict[str, Any]]:
    """
    다중 직접 타깃(Direct Multi-Horizon Target) 기반 Huber 손실 앙상블 모델로 주가를 예측합니다.
    - 1D(내일, t+1), 1W(1주 뒤, t+5), 1M(1달 뒤, t+20) 목표 시점을 직접 회귀
    - Huber Loss (epsilon=1.35) + L2 Ridge Regularization (alpha=15.0) 결합 (수축 및 이상치 저항)
    - 하드코딩 땜질(np.clip 강제 제한, if문 상한선) 완전 배제
    - 인위적 난수(np.random.normal) 없는 순수 결정론적 통계 추론
    - 126 영업일 매끄러운 수렴 시계열 CSV 생성
    """
    clean_code = str(stock_code).replace(".KS", "").zfill(6)
    stock_name = STOCK_INFO.get(clean_code, clean_code)

    # 1. 데이터 확보 (DB 우선, 없을 시 data/ CSV 참조)
    if df is None or df.empty:
        raw_history = get_daily_prices(clean_code)
        if raw_history and len(raw_history) >= 20:
            df = pd.DataFrame(raw_history)
            df.rename(columns={"date": "Date", "close_price": "Close"}, inplace=True)
        else:
            candidates = [
                os.path.join(data_folder, f"{clean_code}_stock_data.csv"),
                os.path.join(data_folder, f"{clean_code}.KS_stock_data.csv"),
                os.path.join(data_folder, f"{stock_name}_stock_data.csv"),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    try:
                        df = pd.read_csv(cand)
                        break
                    except Exception:
                        pass

    if df is None or df.empty or len(df) < 20:
        return None

    df = df.copy()
    if "Date" not in df.columns and "date" in df.columns:
        df["Date"] = df["date"]
    if "Close" not in df.columns and "close_price" in df.columns:
        df["Close"] = df["close_price"]

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 2. 피처 엔지니어링 (누수 정렬)
    feat_df = extract_features(df)
    current_price = float(df["Close"].iloc[-1])
    base_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")

    # 최신 t 시점 피처 (미래 추론용)
    latest_feat = feat_df.iloc[-1:][FEATURE_COLS]

    # 3. 다중 직접 타깃(Direct Multi-Horizon Target) 모델 학습 및 추론
    # 1D: k=1, 1W: k=5, 1M: k=20
    horizons = {"1D": 1, "1W": 5, "1M": 20}
    pred_targets: Dict[str, float] = {}
    pred_log_returns: Dict[str, float] = {}

    for h_name, k in horizons.items():
        # y_k: t 시점 대비 t+k 미래 시점의 로그 수익률 (정확한 시점 정렬)
        y = np.log(feat_df["Close"].shift(-k) / feat_df["Close"])

        valid_mask = y.notna() & feat_df[FEATURE_COLS].notna().all(axis=1)
        X_train = feat_df.loc[valid_mask, FEATURE_COLS]
        y_train = y.loc[valid_mask]

        if len(X_train) < 15:
            # 데이터 극소수 시 평균 보존
            pred_targets[h_name] = current_price
            pred_log_returns[h_name] = 0.0
            continue

        # 피처 표준화 (StandardScaler)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        latest_scaled = scaler.transform(latest_feat)

        # 1) HuberRegressor (이상치 둔감 손실, epsilon=1.35, L2 alpha=15.0)
        huber = HuberRegressor(epsilon=1.35, alpha=15.0, max_iter=1000)
        huber.fit(X_train_scaled, y_train)

        # 2) Ridge Regressor (L2 수축, alpha=15.0)
        ridge = Ridge(alpha=15.0)
        ridge.fit(X_train_scaled, y_train)

        # 3) Huber 앙상블 (Huber 70% + Ridge 30% 가중 평균)
        pred_huber = float(huber.predict(latest_scaled)[0])
        pred_ridge = float(ridge.predict(latest_scaled)[0])
        blended_log_ret = 0.70 * pred_huber + 0.30 * pred_ridge

        pred_price = current_price * np.exp(blended_log_ret)
        pred_targets[h_name] = float(pred_price)
        pred_log_returns[h_name] = float(blended_log_ret)

    pred_1d = float(round(pred_targets["1D"]))
    pred_1w = float(round(pred_targets["1W"]))
    pred_1m = float(round(pred_targets["1M"]))

    # 4. 126 영업일(freq='B') 순수 결정론적(Deterministic) 평활 시계열 생성
    # - 인위적 난수(np.random.normal) 완전 제거
    # - 1일(인덱스 0)=pred_1d, 5일(인덱스 4)=pred_1w, 20일(인덱스 19)=pred_1m 정확 통과
    # - 20일 이후는 Ornstein-Uhlenbeck형 평균 회귀 감쇠로 안정적 장기 수렴
    last_date = df["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=126, freq="B")

    p1 = pred_targets["1D"]
    p5 = pred_targets["1W"]
    p20 = pred_targets["1M"]

    # 최근 60일 일일 평균 로그수익률 (장기 추세 드리프트)
    recent_rets = np.log(df["Close"] / df["Close"].shift(1)).dropna().tail(60)
    mu_daily = float(recent_rets.mean()) if len(recent_rets) > 0 else 0.0
    # 장기 감쇠율 kappa (발산 방지 및 장기 균형 수렴)
    kappa = 0.05

    forecast_records: List[Dict[str, Any]] = []
    for step, f_date in enumerate(future_dates, start=1):
        if step == 1:
            p = p1
        elif step <= 5:
            # 1일 ~ 5일 사이 로그 선형 보간
            p = p1 * np.exp(((step - 1) / 4.0) * np.log(p5 / p1))
        elif step <= 20:
            # 5일 ~ 20일 사이 로그 선형 보간
            p = p5 * np.exp(((step - 5) / 15.0) * np.log(p20 / p5))
        else:
            # 20일 이후 감쇠형 평균 회귀 궤적
            decay_factor = (1.0 - np.exp(-kappa * (step - 20))) / kappa
            p = p20 * np.exp(mu_daily * decay_factor)

        forecast_records.append({
            "Name": clean_code,
            "Date": f_date.strftime("%Y-%m-%d"),
            "Close": round(float(p), 1)
        })

    predicted_df = pd.DataFrame(forecast_records)

    # 5. predict 폴더에 CSV 저장 (차트 및 서빙용 동기화)
    if save_csv:
        output_path_code = os.path.join(predict_folder, f"{clean_code}_predicted_stock_data.csv")
        output_path_ks = os.path.join(predict_folder, f"{clean_code}.KS_predicted_stock_data.csv")
        predicted_df.to_csv(output_path_code, index=False, encoding="utf-8-sig")
        predicted_df.to_csv(output_path_ks, index=False, encoding="utf-8-sig")

        if stock_name and stock_name != clean_code:
            output_path_name = os.path.join(predict_folder, f"{stock_name}_predicted_stock_data.csv")
            df_named = predicted_df.copy()
            df_named["Name"] = stock_name
            df_named.to_csv(output_path_name, index=False, encoding="utf-8-sig")

    # 6. 주요 시점 추출 및 등락률 계산
    rec_1d = forecast_records[0]
    rec_1w = forecast_records[4]
    rec_1m = forecast_records[19]

    date_1d = rec_1d["Date"]
    date_1w = rec_1w["Date"]
    date_1m = rec_1m["Date"]

    change_1d_pct = round(((pred_1d - current_price) / current_price) * 100, 2)
    change_1w_pct = round(((pred_1w - current_price) / current_price) * 100, 2)
    change_1m_pct = round(((pred_1m - current_price) / current_price) * 100, 2)

    return {
        "stock_code": clean_code,
        "stock_name": stock_name,
        "base_date": base_date,
        "current_price": current_price,
        "pred_1d": pred_1d,
        "pred_1w": pred_1w,
        "pred_1m": pred_1m,
        "date_1d": date_1d,
        "date_1w": date_1w,
        "date_1m": date_1m,
        "change_1d_pct": change_1d_pct,
        "change_1w_pct": change_1w_pct,
        "change_1m_pct": change_1m_pct,
        "forecast_df": predicted_df,
        "forecast_series": forecast_records
    }


def run_predict_pipeline() -> Dict[str, Dict[str, Any]]:
    """
    모든 종목에 대해 Huber 손실 직접 타깃(Direct Target) 예측을 수행하고
    stock_prediction 테이블 및 predict/ 폴더 CSV를 일괄 갱신합니다.
    """
    print("================================================================================")
    print("Huber 손실 직접 타깃(Direct Multi-Horizon Target) 예측 파이프라인 시작")
    print(f"대상 종목 수: {len(STOCK_INFO)}개 | 저장 폴더: {os.path.abspath(predict_folder)}")
    print("================================================================================\n")

    results = {}
    stats_1d = []
    stats_1w = []
    stats_1m = []

    for code, name in STOCK_INFO.items():
        try:
            res = predict_stock(code, save_csv=True)
            if res:
                save_prediction(code, res["base_date"], res["pred_1d"], res["pred_1w"], res["pred_1m"])
                results[code] = res
                stats_1d.append(res["change_1d_pct"])
                stats_1w.append(res["change_1w_pct"])
                stats_1m.append(res["change_1m_pct"])
                print(f"-> [{code} {name}] 예측 완료: "
                      f"1D {res['pred_1d']:,.0f} ({res['change_1d_pct']:+.2f}%), "
                      f"1W {res['pred_1w']:,.0f} ({res['change_1w_pct']:+.2f}%), "
                      f"1M {res['pred_1m']:,.0f} ({res['change_1m_pct']:+.2f}%)")
        except Exception as e:
            print(f"-> [!] [{code} {name}] 예측 실패: {e}")

    print("\n================================================================================")
    print(f"총 {len(results)}개 종목의 Huber Direct 예측 및 DB/CSV 갱신이 완료되었습니다.")
    if stats_1d:
        print("--------------------------------------------------------------------------------")
        print("※ 50개 종목 예측 등락률 요약 통계 (발산 여부 검증):")
        print(f"  [1D 내일]   최소: {min(stats_1d):+.2f}% | 최대: {max(stats_1d):+.2f}% | 평균: {np.mean(stats_1d):+.2f}% | 중앙값: {np.median(stats_1d):+.2f}%")
        print(f"  [1W 1주일] 최소: {min(stats_1w):+.2f}% | 최대: {max(stats_1w):+.2f}% | 평균: {np.mean(stats_1w):+.2f}% | 중앙값: {np.median(stats_1w):+.2f}%")
        print(f"  [1M 1달]   최소: {min(stats_1m):+.2f}% | 최대: {max(stats_1m):+.2f}% | 평균: {np.mean(stats_1m):+.2f}% | 중앙값: {np.median(stats_1m):+.2f}%")
        print("--------------------------------------------------------------------------------")
    print("================================================================================")
    return results


if __name__ == "__main__":
    # 전체 종목 대상 파이프라인 실행
    run_predict_pipeline()