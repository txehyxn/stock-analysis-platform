-- 일별 시세 테이블 (종가 중심)
CREATE TABLE IF NOT EXISTS stock_daily_price (
    stock_code VARCHAR(20) NOT NULL,
    date VARCHAR(10) NOT NULL,
    close_price REAL NOT NULL,
    PRIMARY KEY (stock_code, date)
);

-- 예측 결과 테이블 (1일, 1주일(5일), 1달(20일) 예측치)
CREATE TABLE IF NOT EXISTS stock_prediction (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code VARCHAR(20) NOT NULL,
    base_date VARCHAR(10) NOT NULL,
    pred_1d REAL NOT NULL,
    pred_1w REAL NOT NULL,
    pred_1m REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_daily_price_code_date ON stock_daily_price (stock_code, date);
CREATE INDEX IF NOT EXISTS idx_prediction_code_date ON stock_prediction (stock_code, base_date);
