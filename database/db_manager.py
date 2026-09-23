import os
import sqlite3
from typing import List, Dict, Any, Optional
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_data.db")
DATABASE_URL = os.getenv("DATABASE_URL", None)

STOCK_INFO = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "373220": "LG에너지솔루션",
    "207940": "삼성바이오로직스",
    "005380": "현대차",
    "000270": "기아",
    "068270": "셀트리온",
    "105560": "KB금융",
    "035420": "NAVER",
    "055550": "신한지주",
    "005490": "POSCO홀딩스",
    "012330": "현대모비스",
    "028260": "삼성물산",
    "035720": "카카오",
    "006400": "삼성SDI",
    "051910": "LG화학",
    "086790": "하나금융지주",
    "138040": "메리츠금융지주",
    "011200": "HMM",
    "010130": "고려아연",
    "033780": "KT&G",
    "032830": "삼성생명",
    "259960": "크래프톤",
    "012450": "한화에어로스페이스",
    "003670": "포스코퓨처엠",
    "323410": "카카오뱅크",
    "015760": "한국전력",
    "450080": "에코프로머티",
    "030200": "KT",
    "096770": "SK이노베이션",
    "402340": "SK스퀘어",
    "034020": "두산에너빌리티",
    "352820": "하이브",
    "041510": "에스엠",
    "329180": "HD현대중공업",
    "267260": "HD현대일렉트릭",
    "316140": "우리금융지주",
    "066570": "LG전자",
    "247540": "에코프로비엠",
    "086520": "에코프로",
    "028300": "HLB",
    "196170": "알테오젠",
    "036570": "엔씨소프트",
    "251270": "넷마블",
    "263750": "펄어비스",
    "042700": "한미반도체",
    "058470": "리노공업",
    "277810": "레인보우로보틱스",
    "328130": "루닛",
    "035900": "JYP Ent."
}

def get_connection():
    """데이터베이스 연결 객체를 반환합니다. (기본 SQLite)"""
    if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
        try:
            import psycopg2
            return psycopg2.connect(DATABASE_URL)
        except ImportError:
            print("Warning: psycopg2 not installed, falling back to SQLite.")
    
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """데이터베이스 스키마를 초기화합니다."""
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    
    conn = get_connection()
    try:
        if hasattr(conn, "executescript"):
            conn.executescript(schema_sql)
        else:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
        conn.commit()
    finally:
        conn.close()

def save_daily_prices(df: pd.DataFrame) -> int:
    """
    일별 시세 데이터프레임(stock_code, date, close_price)을 데이터베이스에 UPSERT 일괄(executemany) 적재합니다.
    """
    if df.empty:
        return 0

    init_db()
    conn = get_connection()
    inserted_count = 0

    rows_to_insert = []
    for _, row in df.iterrows():
        stock_code = str(row["stock_code"]).zfill(6)
        date_val = str(row["date"]).strip()
        close_price = float(row["close_price"])
        rows_to_insert.append((stock_code, date_val, close_price))

    try:
        cursor = conn.cursor()
        if isinstance(conn, sqlite3.Connection):
            cursor.executemany("""
                INSERT INTO stock_daily_price (stock_code, date, close_price)
                VALUES (?, ?, ?)
                ON CONFLICT(stock_code, date) DO UPDATE SET close_price=excluded.close_price
            """, rows_to_insert)
        else:
            # PostgreSQL
            cursor.executemany("""
                INSERT INTO stock_daily_price (stock_code, date, close_price)
                VALUES (%s, %s, %s)
                ON CONFLICT (stock_code, date) DO UPDATE SET close_price = EXCLUDED.close_price
            """, rows_to_insert)
        
        conn.commit()
        inserted_count = len(rows_to_insert)
    finally:
        conn.close()

    return inserted_count

def save_prediction(stock_code: str, base_date: str, pred_1d: float, pred_1w: float, pred_1m: float) -> int:
    """
    예측 결과를 stock_prediction 테이블에 저장합니다.
    """
    init_db()
    conn = get_connection()
    stock_code = str(stock_code).zfill(6)
    try:
        cursor = conn.cursor()
        if isinstance(conn, sqlite3.Connection):
            cursor.execute("""
                INSERT INTO stock_prediction (stock_code, base_date, pred_1d, pred_1w, pred_1m)
                VALUES (?, ?, ?, ?, ?)
            """, (stock_code, base_date, float(pred_1d), float(pred_1w), float(pred_1m)))
            pred_id = cursor.lastrowid
        else:
            cursor.execute("""
                INSERT INTO stock_prediction (stock_code, base_date, pred_1d, pred_1w, pred_1m)
                VALUES (%s, %s, %s, %s, %s) RETURNING id
            """, (stock_code, base_date, float(pred_1d), float(pred_1w), float(pred_1m)))
            pred_id = cursor.fetchone()[0]
        conn.commit()
        return pred_id
    finally:
        conn.close()

def get_daily_prices(stock_code: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    특정 종목의 일별 시세 목록(날짜 오름차순)을 반환합니다.
    """
    init_db()
    conn = get_connection()
    stock_code = str(stock_code).zfill(6)
    try:
        cursor = conn.cursor()
        query = """
            SELECT stock_code, date, close_price 
            FROM stock_daily_price 
            WHERE stock_code = ? 
            ORDER BY date ASC
        """
        if limit:
            query = f"SELECT * FROM ({query} DESC LIMIT {limit}) ORDER BY date ASC"

        if isinstance(conn, sqlite3.Connection):
            cursor.execute(query, (stock_code,))
        else:
            query_pg = query.replace("?", "%s")
            cursor.execute(query_pg, (stock_code,))

        rows = cursor.fetchall()
        result = []
        for r in rows:
            if isinstance(r, sqlite3.Row):
                result.append(dict(r))
            else:
                result.append({
                    "stock_code": r[0],
                    "date": r[1],
                    "close_price": r[2]
                })
        return result
    finally:
        conn.close()

def get_latest_prediction(stock_code: str) -> Optional[Dict[str, Any]]:
    """
    특정 종목의 가장 최근 예측 결과를 반환합니다.
    """
    init_db()
    conn = get_connection()
    stock_code = str(stock_code).zfill(6)
    try:
        cursor = conn.cursor()
        query = """
            SELECT id, stock_code, base_date, pred_1d, pred_1w, pred_1m, created_at 
            FROM stock_prediction 
            WHERE stock_code = ? 
            ORDER BY id DESC LIMIT 1
        """
        if isinstance(conn, sqlite3.Connection):
            cursor.execute(query, (stock_code,))
        else:
            query_pg = query.replace("?", "%s")
            cursor.execute(query_pg, (stock_code,))

        row = cursor.fetchone()
        if not row:
            return None

        if isinstance(row, sqlite3.Row):
            return dict(row)
        else:
            return {
                "id": row[0],
                "stock_code": row[1],
                "base_date": row[2],
                "pred_1d": row[3],
                "pred_1w": row[4],
                "pred_1m": row[5],
                "created_at": row[6]
            }
    finally:
        conn.close()

def get_stock_list() -> List[Dict[str, str]]:
    """지원 종목 리스트를 반환합니다."""
    return [{"code": code, "name": name} for code, name in STOCK_INFO.items()]
