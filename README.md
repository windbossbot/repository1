# Zero-Key KRW Screener CLI

무료 데이터 소스만 사용해 KOSPI / NASDAQ / CRYPTO를 스크리닝하는 CLI입니다.

## 설치
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 사용법
```bash
python -m src.main reset
python -m src.main page --page 1
python -m src.main next
```

출력은 설명 없이 JSON만 출력합니다.

## 캐시/상태 파일
- `data/cache_fx.json` : USDKRW 환율 캐시
- `data/cache_symbols_nasdaq.txt` : NasdaqTrader 심볼 캐시
- `data/cache_coingecko_symbols.json` : 코인 심볼 매핑 캐시
- `data/state.json` : 페이지 커서 상태
