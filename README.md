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

## Cloudflare Worker API
`worker.js`는 Binance klines 기반의 **방향성 중립(stress-only)** 시장 스트레스 점수를 계산하는 Cloudflare Worker입니다.

### 요청 형식
```json
{
  "symbol": "BTCUSDT",
  "interval": "1m"
}
```

### 응답 형식
```json
{
  "symbol": "BTCUSDT",
  "stress_score": 0.42,
  "stress_level": "MEDIUM",
  "confidence": 0.81,
  "directional_tilt": {
    "bias": "UP",
    "tilt_strength": 0.37
  }
}
```

- `stress_score`는 **절대수익률 변동성 + 거래량 스파이크 비율**로 계산되는 0~1 스코어입니다.
- `directional_tilt`는 참고용(예측 아님)으로, 120기간 이동평균 대비 괴리에서 계산됩니다.
- 브라우저 호출을 위해 CORS(`POST, OPTIONS`)가 활성화되어 있습니다.

---

## 클라우드플레어(Cloudflare) 배포 방법
질문하신 화면(Ship something new)에서는 `Start with Hello World!`로 시작하는 게 가장 쉽습니다.

### 방법 A) Dashboard에서 바로 배포 (가장 간단)
1. Cloudflare Dashboard → **Workers & Pages** → **Create application**.
2. 질문 이미지 화면에서 **Start with Hello World!** 클릭.
3. Worker 이름 입력 후 생성.
4. 기본 코드를 지우고, 이 저장소의 `worker.js` 내용을 그대로 붙여넣기.
5. **Deploy** 클릭.
6. 배포 후 발급되는 URL(예: `https://<worker-name>.<subdomain>.workers.dev`)로 호출.

테스트 예시:
```bash
curl -X POST "https://<worker-url>" \
  -H "content-type: application/json" \
  -d '{"symbol":"BTCUSDT","interval":"1m"}'
```

### 방법 B) GitHub 연결 자동 배포
1. 같은 화면에서 **Connect GitHub** 클릭.
2. 이 저장소 연결 후 Worker 프로젝트 생성.
3. Root에 있는 `wrangler.toml`과 `worker.js` 기준으로 빌드/배포.
4. 브랜치 푸시 시 자동 배포 설정 가능.

### 방법 C) 로컬 CLI(wrangler) 배포
```bash
npm install -g wrangler
wrangler login
wrangler deploy
```

### 배포 시 주의사항
- Binance API는 지역/네트워크 정책으로 간헐 차단될 수 있어 `502`가 나올 수 있습니다.
- 현재 Worker는 입력 오류/업스트림 오류를 안전한 JSON으로 반환하도록 처리되어 있습니다.
