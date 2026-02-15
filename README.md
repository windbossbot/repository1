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

## Render 배포 (Blueprint)
로컬 Python 설치 없이 Render에서 바로 배포할 수 있습니다.

1. GitHub 저장소를 Render에 연결하고 **New + > Blueprint**를 선택합니다.
2. 저장소 루트의 `render.yaml`을 사용해 웹 서비스를 생성합니다.
3. Render가 `pip install -r requirements.txt`로 의존성을 자동 설치합니다.
4. 서비스는 `streamlit run coin_app.py --server.port $PORT --server.address 0.0.0.0`로 실행되어 `$PORT`와 `0.0.0.0`에 바인딩됩니다.

