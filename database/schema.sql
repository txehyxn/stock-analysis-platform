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

-- 사용자 계정 테이블
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    username TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 사용자 관심 종목(즐겨찾기) 테이블
CREATE TABLE IF NOT EXISTS user_favorites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    stock_code VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE (user_id, stock_code)
);

CREATE INDEX IF NOT EXISTS idx_user_favorites_user ON user_favorites (user_id);

-- 실전 예측 자동 채점 및 로그 테이블
CREATE TABLE IF NOT EXISTS prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code VARCHAR(10) NOT NULL,
    predicted_at DATE NOT NULL,
    target_date DATE NOT NULL,
    period_type VARCHAR(5) NOT NULL, -- '1D' 또는 '1W'
    base_price INTEGER NOT NULL,
    predicted_price INTEGER NOT NULL,
    actual_price INTEGER,
    is_hit BOOLEAN,
    error_rate REAL,
    status VARCHAR(10) DEFAULT 'PENDING', -- 'PENDING' 또는 'EVALUATED'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(stock_code, predicted_at, period_type)
);

CREATE INDEX IF NOT EXISTS idx_pred_logs_stock_target ON prediction_logs (stock_code, target_date);
CREATE INDEX IF NOT EXISTS idx_pred_logs_status ON prediction_logs (status);
