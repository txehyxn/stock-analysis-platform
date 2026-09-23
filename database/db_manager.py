import os
import sqlite3
from typing import List, Dict, Any, Optional
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_data.db")
DATABASE_URL = os.getenv("DATABASE_URL", None)

STOCK_INFO = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "005380": "현대차",
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
    일별 시세 데이터프레임(stock_code, date, close_price)을 데이터베이스에 UPSERT 적재합니다.
    """
    if df.empty:
        return 0

    init_db()
    conn = get_connection()
    inserted_count = 0

    try:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            stock_code = str(row["stock_code"]).zfill(6)
            date_val = str(row["date"]).strip()
            close_price = float(row["close_price"])

            # SQLite의 INSERT OR REPLACE
            if isinstance(conn, sqlite3.Connection):
                cursor.execute("""
                    INSERT INTO stock_daily_price (stock_code, date, close_price)
                    VALUES (?, ?, ?)
                    ON CONFLICT(stock_code, date) DO UPDATE SET close_price=excluded.close_price
                """, (stock_code, date_val, close_price))
            else:
                # PostgreSQL
                cursor.execute("""
                    INSERT INTO stock_daily_price (stock_code, date, close_price)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (stock_code, date) DO UPDATE SET close_price = EXCLUDED.close_price
                """, (stock_code, date_val, close_price))
            inserted_count += 1

        conn.commit()
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
