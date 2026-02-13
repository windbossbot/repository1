# Local 1-Page Market Regime Analyzer

FastAPI 기반 로컬 웹 앱입니다. **실시간 스트리밍 없이** 버튼 클릭 시에만 데이터를 조회하고, 동일 요청은 로컬 캐시(기본 12시간 TTL)를 사용합니다.

## 기능
- [분석] 버튼
  - 시장 국면 판정(주봉, KOSPI/NASDAQ/BTC/ETH)
  - Fear & Greed Index 표시
  - 소스 실패 시 앱 종료 없이 오류 표시
- [강세장 스크리너] / [전환·약세 스크리너]
  - 조건 기반 테이블 출력
- 1페이지 UI (숫자/상태 중심)

## 데이터 소스
- KOSPI: stooq CSV (`^KOSPI`)
- NASDAQ: stooq CSV 직접 조회 (`^NDQ`, fallback 심볼 포함)
- BTC/ETH: `ccxt` + Kraken 우선, 실패 시 Coinbase로 자동 fallback (`BTC/USD`, `ETH/USD`)
- Fear & Greed: `https://api.alternative.me/fng/?limit=1`

## 실행
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

브라우저: http://127.0.0.1:8000

## 캐시
- 경로: `./cache/cache.sqlite3`
- TTL: 12시간
- 키: 요청 파라미터 기반 SHA-256

## 아무것도 안 나올 때 체크
1. 브라우저에서 `http://127.0.0.1:8000/api/ping` 접속 시 `{"status":"ok"}` 가 나오는지 확인
2. 터미널에서 uvicorn 실행 로그에 import 에러가 없는지 확인
3. 상단 상태가 `서버 연결 실패`면 주소/포트를 다시 확인 (`127.0.0.1:8000`)
4. 분석/스크리너는 버튼 클릭 전까지 결과가 비어있는 것이 정상

## 참고
- 스크리너 KRX 종목군은 실행 시간을 줄이기 위해 상위 시가총액 종목 일부를 대상으로 검사합니다.


## Render 배포 시 참고
- `gunicorn: command not found` 오류가 나면 `requirements.txt`에 `gunicorn`이 포함되어 있는지 확인하세요.
- Procfile은 다음처럼 `web` 프로세스로 실행해야 합니다.

```
web: gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT:-8000}
```

- pandas 버전은 Render에서 안정적으로 동작하는 버전으로 고정했습니다. (pandas-datareader 미사용)
