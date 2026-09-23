# Stock Analysis & 1D-CNN Prediction Platform

AI 기반 네이버 증권 데이터 자동 수집, SQLite 적재, PyTorch 1D-CNN 다중 회귀(Multi-output Regression) 주가 예측 및 모바일 최적화 웹 대시보드 플랫폼입니다.

---

## 🚀 주요 기능 및 아키텍처

1. **스마트 크롤러 (`crawler/stock_spider.py`)**
   - 네이버 증권 일별 시세에서 삼성전자(`005930`), SK하이닉스(`000660`), 현대차(`005380`) 등 주요 종목 수집.
   - 불필요한 데이터를 제외하고 핵심 3개 컬럼(`stock_code`, `date`, `close`)만 추출.
   - 날짜 오름차순 정렬 후 `data/excel/stock_{code}.xlsx`로 엑셀 백업 및 SQLite DB 적재.

2. **데이터베이스 레이어 (`database/schema.sql`, `database/db_manager.py`)**
   - `stock_daily_price`: 종목별 일별 종가 저장 (PK: `stock_code`, `date`)
   - `stock_prediction`: 종목별 최신 1D-CNN 예측치(1일 뒤, 1주 뒤, 1달 뒤) 저장
   - SQLite(`stock_data.db`) 및 Docker PostgreSQL 환경 확장 지원.

3. **PyTorch 1D-CNN 머신러닝 예측기 (`ml_model/train_cnn.py`)**
   - 과거 30영업일 종가 시계열 입력 -> 미래 3개 시점([1일 뒤, 5일 뒤(1주), 20일 뒤(1달)]) 종가 다중 회귀 추정.
   - Conv1d + BatchNorm1d + ReLU + MaxPool1d + Regressor 구조.
   - 산출된 미래 예측치 DB 테이블 `stock_prediction` 적재.

4. **모바일 반응형 웹 대시보드 (`web/app.py`, `web/templates/index.html`)**
   - FastAPI 비동기 백엔드 (`/api/stocks`, `/api/stock/{code}`).
   - Tailwind CSS 다크모드, 반응형 Viewport 메타태그, 글래스모피즘 UI.
   - Chart.js 시각화: 실제 종가(파란색 실선) + 1D/1W/1M 예측 트렌드(주황색 점선).
   - 모바일 세로 화면 1열 자동 전환 요약 카드.

---

## 🛠️ 실행 방법

### 1. 가상환경 및 패키지 설치
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 데이터 크롤링 및 엑셀/DB 적재
```bash
.\.venv\Scripts\python.exe crawler/stock_spider.py
```

### 3. 1D-CNN 모델 학습 및 예측값 생성
```bash
.\.venv\Scripts\python.exe ml_model/train_cnn.py
```

### 4. 웹 대시보드 서버 구동
```bash
.\.venv\Scripts\python.exe web/app.py
```
- 브라우저 접속: `http://localhost:8000` (모바일 브라우저 및 외부 접속 가능)
