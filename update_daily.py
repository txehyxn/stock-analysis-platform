import os
import sys
from datetime import datetime

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

# 프로젝트 루트 모듈 참조
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from crawler.stock_spider import crawl_and_store_all
from predict import predict_stock
from database.db_manager import (
    evaluate_pending_predictions,
    save_prediction_log,
    save_prediction,
    get_evaluation_summary,
    get_connection,
    STOCK_INFO,
    init_db
)

def run_daily_pipeline(today_date: str = None) -> dict:
    """
    일일 시세 갱신, 과거 예측 자동 채점 및 predict.py(Huber Direct) 기반 신규 예측 로그 생성을 순차 실행합니다.
    """
    if not today_date:
        today_date = datetime.now().strftime("%Y-%m-%d")

    init_db()

    print("\n" + "=" * 70)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [*] 일일 주식 시세 갱신 & Huber Direct AI 예측 채점 파이프라인 시작 (기준일: {today_date})")
    print("=" * 70)

    # 1. 50개 전 종목 일별 시세 크롤링 및 DB 적재
    print("\n[단계 1/3] 50개 전 종목 최신 시세 수집 중...")
    crawl_and_store_all(target_days=120)

    # 2. 과거 PENDING 상태 예측들에 대한 자동 채점 수행
    print(f"\n[단계 2/3] 도래일({today_date} 이전/당일) 미채점 예측 건 자동 평가 및 채점...")
    eval_count = evaluate_pending_predictions(today_date)
    print(f"  -> 총 {eval_count}건의 실전 예측 채점 완료 (EVALUATED 처리)")

    # 3. 50개 전 종목 predict.py (Huber Loss 다중 직접 타깃) 예측 모델 실행 및 신규 예측 로그 저장
    print("\n[단계 3/3] predict.py (Huber Direct 다중 직접 타깃) 예측 모델 실행 및 실제 CSV 생성/적재...")
    
    new_logs_count = 0
    success_stocks_count = 0

    for code, name in STOCK_INFO.items():
        try:
            # predict.py의 RandomForest 알고리즘 수행 및 predict/{code}_predicted_stock_data.csv 생성
            res = predict_stock(code, save_csv=True)
            if not res:
                print(f"  [!] [{code} {name}] 예측 데이터 부족으로 스킵")
                continue

            base_date = res["base_date"]
            current_price = int(round(res["current_price"]))
            pred_1d = int(round(res["pred_1d"]))
            pred_1w = int(round(res["pred_1w"]))
            pred_1m = int(round(res["pred_1m"]))

            target_1d = res["date_1d"]
            target_1w = res["date_1w"]
            target_1m = res["date_1m"]

            # stock_prediction 테이블 적재
            save_prediction(code, base_date, pred_1d, pred_1w, pred_1m)

            # 1D 예측 로그 저장
            save_prediction_log(
                stock_code=code,
                predicted_at=base_date,
                target_date=target_1d,
                period_type="1D",
                base_price=current_price,
                predicted_price=pred_1d,
                status="PENDING"
            )

            # 1W 예측 로그 저장
            save_prediction_log(
                stock_code=code,
                predicted_at=base_date,
                target_date=target_1w,
                period_type="1W",
                base_price=current_price,
                predicted_price=pred_1w,
                status="PENDING"
            )

            # 1M 예측 로그 저장
            save_prediction_log(
                stock_code=code,
                predicted_at=base_date,
                target_date=target_1m,
                period_type="1M",
                base_price=current_price,
                predicted_price=pred_1m,
                status="PENDING"
            )

            new_logs_count += 3
            success_stocks_count += 1
            print(f"  [v] [{code} {name}] 예측 완료: 1D {pred_1d:,}원 ({res['change_1d_pct']:+.2f}%), 1W {pred_1w:,}원 ({res['change_1w_pct']:+.2f}%), 1M {pred_1m:,}원 ({res['change_1m_pct']:+.2f}%)")

        except Exception as e:
            print(f"  [x] [{code} {name}] 처리 중 오류: {e}")

    # 최종 채점 통계 확인
    summary = get_evaluation_summary()

    # DB의 실제 최신 일자 확인
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM stock_daily_price")
        row = cursor.fetchone()
        latest_date_in_db = row[0] if row else today_date
        conn.close()
    except Exception:
        latest_date_in_db = today_date

    completion_date = latest_date_in_db or today_date

    print("\n" + "=" * 70)
    print("  [v] 일일 파이프라인 및 Huber Direct AI 예측 채점 완료 요약:")
    print(f"      - DB 최신 기준일자: {completion_date}")
    print(f"      - 성공 종목 수: {success_stocks_count} / {len(STOCK_INFO)}")
    print(f"      - 평가 완료 건수: {summary['total_count']:,}건")
    print(f"      - 실전 누적 적중률: {summary['hit_rate']}% ({summary['hit_count']}/{summary['total_count'] if summary['total_count'] > 0 else 1})")
    print(f"      - 평균 주가 오차율: ±{summary['avg_error_rate']}%")
    print(f"      - 신규 등록 예측로그: {new_logs_count}건 (대기 중: {summary['pending_count']}건)")
    print("=" * 70)
    print(f"Daily Stock & Prediction Update Completed for {completion_date}\n")

    return {
        "status": "success",
        "completion_date": completion_date,
        "evaluated_count": eval_count,
        "new_logs_count": new_logs_count,
        "summary": summary
    }

if __name__ == "__main__":
    param_date = sys.argv[1] if len(sys.argv) > 1 else None
    run_daily_pipeline(param_date)
