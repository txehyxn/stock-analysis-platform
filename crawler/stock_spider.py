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

def fetch_stock_daily_prices(stock_code: str, pages: int = 10, page_size: int = 20) -> pd.DataFrame:
    """
    네이버 증권에서 특정 종목의 일별 시세를 크롤링합니다.
    오직 'stock_code', 'date', 'close' 컬럼만 추출하고 날짜 오름차순으로 정렬합니다.
    """
    records = []
    stock_name = STOCK_INFO.get(stock_code, stock_code)
    print(f"[*] Crawling {stock_name} ({stock_code}) - {pages} pages ({pages * page_size} days max)...")

    # 1. 네이버 증권 모바일 API 우선 시도
    for page in range(1, pages + 1):
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
                        # 콤마 제거 및 float 변환
                        close_clean = float(str(close_raw).replace(",", ""))
                        records.append({
                            "stock_code": stock_code,
                            "date": date_val,
                            "close": close_clean
                        })
                time.sleep(0.05)
            else:
                print(f"[-] API status {resp.status_code} on page {page}")
        except Exception as e:
            print(f"[-] Error fetching API on page {page}: {e}")

    # Fallback: 만약 API 응답이 없으면 레거시 HTML 파싱 시도
    if not records:
        print("[*] Trying fallback HTML parser...")
        for page in range(1, pages + 1):
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
                                        "stock_code": stock_code,
                                        "date": d_txt,
                                        "close": float(c_txt)
                                    })
                                except ValueError:
                                    pass
            except Exception as e:
                print(f"[-] Fallback error: {e}")

    if not records:
        print(f"[-] No records found for {stock_code}")
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
    print(f"[+] Saved Excel backup to {file_path} ({len(df)} rows)")
    return file_path

def crawl_and_store_all(pages: int = 10, page_size: int = 20):
    """
    모든 대상 종목(삼성전자, SK하이닉스, 현대차)을 크롤링하고
    1) 엑셀 백업 저장 (stock_code, date, close)
    2) 데이터베이스(stock_daily_price)에 적재합니다.
    """
    init_db()
    total_loaded = 0

    for code in STOCK_INFO.keys():
        df = fetch_stock_daily_prices(code, pages=pages, page_size=page_size)
        if df.empty:
            continue

        # 1. 엑셀 백업 저장
        save_to_excel(df, code)

        # 2. DB 적재 (컬럼명 close -> close_price 매핑)
        db_df = df.rename(columns={"close": "close_price"})
        count = save_daily_prices(db_df)
        total_loaded += count
        print(f"[+] Loaded {count} rows into DB for {code}")

    print(f"\n[v] Completed all crawling and DB loading. Total records updated: {total_loaded}")

if __name__ == "__main__":
    pages = 10
    if len(sys.argv) > 1:
        try:
            pages = int(sys.argv[1])
        except ValueError:
            pass
    crawl_and_store_all(pages=pages)

