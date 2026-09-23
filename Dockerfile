# Python 3.11 Slim 경량 베이스 이미지 사용
FROM python:3.11-slim

# 환경 변수 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 필수 빌드 도구 및 유틸리티 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 및 설치 (캐시 레이어 최적화)
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 코드 복사
COPY . /app/

# 데이터 및 엑셀 디렉토리 생성
RUN mkdir -p /app/data/excel

# 포트 개방
EXPOSE 8000

# 컨테이너 헬스체크 (선택사항)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/stocks || exit 1

# FastAPI 서버 기동
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
