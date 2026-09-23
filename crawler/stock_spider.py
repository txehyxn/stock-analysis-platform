import os
import sys
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

# 상위 디렉토리 모듈 참조
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import save_daily_prices, STOCK_INFO, init_db

API_URL = "https://m.stock.naver.com/api/stock/{code}/price"
HTML_URL = "https://finance.naver.com/item/sise_day.naver"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Referer": "https://m.stock.naver.com"
}
EXCEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "excel")

# 50대 주요 종목 매핑 정의
TARGET_STOCKS = STOCK_INFO

def fetch_stock_daily_prices(stock_code: str, target_days: int = 120) -> pd.DataFrame:
    """
    네이버 증권에서 특정 종목의 최근 약 120영업일 일별 시세를 크롤링합니다.
    오직 'stock_code', 'date', 'close' 3개 컬럼만 추출하고 날짜 오름차순으로 정렬합니다.
    """
    records = []
    page_size = 20
    pages_needed = (target_days + page_size - 1) // page_size # 120일 -> 6페이지 (20개씩)

    for page in range(1, pages_needed + 1):
        url = f"https://m.stock.naver.com/api/stock/{stock_code}/price?pageSize={page_size}&page={page}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if not data or not isinstance(data, list):
                    break
                
                for item in data:
                    date_val = item.get("localTradedAt")
                    close_raw = item.get("closePrice")
                    if date_val and close_raw:
                        close_clean = float(str(close_raw).replace(",", ""))
                        records.append({
                            "stock_code": str(stock_code).zfill(6),
                            "date": date_val,
                            "close": close_clean
                        })
                # 페이지당 0.15초 대기 (차단 방지)
                time.sleep(0.15)
            else:
                time.sleep(0.15)
        except Exception as e:
            time.sleep(0.15)

    # Fallback: API 응답이 없을 경우 HTML 파싱 시도
    if not records:
        for page in range(1, 13):
            url = f"{HTML_URL}?code={stock_code}&page={page}"
            try:
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "lxml")
                    for row in soup.select("table.type2 tr"):
                        cols = row.select("td")
                        if len(cols) >= 7:
                            d_txt = cols[0].get_text(strip=True).replace(".", "-")
                            c_txt = cols[1].get_text(strip=True).replace(",", "")
                            if d_txt and c_txt:
                                try:
                                    records.append({
                                        "stock_code": str(stock_code).zfill(6),
                                        "date": d_txt,
                                        "close": float(c_txt)
                                    })
                                except ValueError:
                                    pass
                time.sleep(0.15)
            except Exception:
                time.sleep(0.15)

    if not records:
        return pd.DataFrame(columns=["stock_code", "date", "close"])

    # DataFrame 생성 및 오직 3개 컬럼만 유지
    df = pd.DataFrame(records)[["stock_code", "date", "close"]]
    # 중복 제거 및 날짜 오름차순 정렬
    df = df.drop_duplicates(subset=["stock_code", "date"]).sort_values(by="date", ascending=True).reset_index(drop=True)
    return df

def save_to_excel(df: pd.DataFrame, stock_code: str) -> str:
    """
    크롤링 데이터를 openpyxl 엔진을 사용하여 data/excel/stock_{code}.xlsx 로 백업 저장합니다.
    """
    os.makedirs(EXCEL_DIR, exist_ok=True)
    file_path = os.path.join(EXCEL_DIR, f"stock_{stock_code}.xlsx")
    df.to_excel(file_path, index=False, engine="openpyxl")
    return file_path

def crawl_and_store_all(target_days: int = 120):
    """
    50개 주요 종목을 순회하며 일별 시세 수집, 엑셀 백업, DB 일괄 적재를 수행합니다.
    """
    init_db()
    total_stocks = len(TARGET_STOCKS)
    success_count = 0
    total_records = 0

    print("=" * 70)
    print(f"[*] 국내 시가총액 상위 {total_stocks}개 종목 시세 수집 시작 (종목당 최근 {target_days}영업일)...")
    print("=" * 70)

    for idx, (code, name) in enumerate(TARGET_STOCKS.items(), 1):
        try:
            # 1. 시세 크롤링
            df = fetch_stock_daily_prices(code, target_days=target_days)
            if df.empty:
                print(f"[{idx:02d}/{total_stocks}] ⚠️  {name}({code}): 수집 데이터 없음")
                continue

            # 2. 엑셀 백업 저장 (stock_code, date, close)
            save_to_excel(df, code)

            # 3. DB 일괄 적재 (executemany)
            db_df = df.rename(columns={"close": "close_price"})
            count = save_daily_prices(db_df)
            total_records += count
            success_count += 1

            print(f"[{idx:02d}/{total_stocks}] [OK] {name}({code}): {count}건 수집 -> 엑셀/DB 적재 완료")

            # 종목 전환 시 0.25초 대기 (네이버 부하 방지)
            time.sleep(0.25)

        except Exception as e:
            print(f"[{idx:02d}/{total_stocks}] [ERR] {name}({code}) 처리 중 오류 발생: {e}")
            time.sleep(0.25)
            continue

    print("=" * 70)
    print(f"[v] 50개 종목 수집 완료: 성공 {success_count}/{total_stocks} 종목 (총 {total_records:,}건 적재)")
    print("=" * 70)

if __name__ == "__main__":
    target_days = 120
    if len(sys.argv) > 1:
        try:
            target_days = int(sys.argv[1])
        except ValueError:
            pass
    crawl_and_store_all(target_days=target_days)

