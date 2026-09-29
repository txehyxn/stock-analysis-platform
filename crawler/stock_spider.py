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
    특정 종목의 최근 약 target_days(기본 120영업일) 일별 시세를 크롤링합니다.
    다음(Daum) 금융 API를 1차로 활용하며, 네이버 증권 API를 폴백으로 사용합니다.
    오직 'stock_code', 'date', 'close' 3개 컬럼만 추출하고 날짜 오름차순으로 정렬합니다.
    """
    records = []
    stock_code_clean = str(stock_code).zfill(6)

    # 1. Daum 금융 일별 시세 API (초고속 및 안정적)
    daum_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Referer": "https://finance.daum.net"
    }
    per_page = 30
    pages_needed = (target_days + per_page - 1) // per_page

    try:
        for page in range(1, pages_needed + 1):
            url = f"https://finance.daum.net/api/quote/A{stock_code_clean}/days?perPage={per_page}&page={page}"
            resp = requests.get(url, headers=daum_headers, timeout=8)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if not data:
                    break
                for item in data:
                    d_raw = item.get("date", "")
                    c_raw = item.get("tradePrice")
                    if d_raw and c_raw is not None:
                        date_str = str(d_raw)[:10] # YYYY-MM-DD
                        records.append({
                            "stock_code": stock_code_clean,
                            "date": date_str,
                            "close": float(c_raw)
                        })
                time.sleep(0.1)
            else:
                break
    except Exception as e:
        print(f"    [Daum API Fallback] {stock_code_clean}: {e}")

    # 2. Fallback: 네이버 증권 모바일 API
    if not records:
        page_size = 20
        naver_pages = (target_days + page_size - 1) // page_size
        for page in range(1, naver_pages + 1):
            url = f"https://m.stock.naver.com/api/stock/{stock_code_clean}/price?pageSize={page_size}&page={page}"
            try:
                resp = requests.get(url, headers=HEADERS, timeout=8)
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
                                "stock_code": stock_code_clean,
                                "date": date_val,
                                "close": close_clean
                            })
                    time.sleep(0.1)
            except Exception:
                time.sleep(0.1)

    # 3. Fallback: HTML 파싱 시도
    if not records:
        for page in range(1, 13):
            url = f"{HTML_URL}?code={stock_code_clean}&page={page}"
            try:
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
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
                                        "stock_code": stock_code_clean,
                                        "date": d_txt,
                                        "close": float(c_txt)
                                    })
                                except ValueError:
                                    pass
                time.sleep(0.1)
            except Exception:
                time.sleep(0.1)

    if not records:
        return pd.DataFrame(columns=["stock_code", "date", "close"])

    # DataFrame 생성 및 중복 제거, 날짜 오름차순 정렬
    df = pd.DataFrame(records)[["stock_code", "date", "close"]]
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

