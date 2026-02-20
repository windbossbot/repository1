const BINANCE_KLINES_URL = 'https://api.binance.com/api/v3/klines';
const MAX_LIMIT = 240;
const MA_PERIOD = 120;
const ALLOWED_INTERVALS = new Set([
  '1m', '3m', '5m', '15m', '30m',
  '1h', '2h', '4h', '6h', '8h', '12h',
  '1d', '3d', '1w', '1M'
]);

const CORS_HEADERS = {
  'access-control-allow-origin': '*',
  'access-control-allow-methods': 'POST, OPTIONS',
  'access-control-allow-headers': 'content-type'
};

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      ...CORS_HEADERS
    }
  });
}

function corsPreflightResponse() {
  return new Response(null, {
    status: 204,
    headers: CORS_HEADERS
  });
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function safeNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function stdDev(values) {
  if (!values.length) return 0;
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function classifyStress(score) {
  if (score >= 0.67) return 'HIGH';
  if (score >= 0.34) return 'MEDIUM';
  return 'LOW';
}

function parseAndValidatePayload(payload) {
  if (!payload || typeof payload !== 'object') {
    return { error: 'Request body must be a JSON object.' };
  }

  const rawSymbol = typeof payload.symbol === 'string' ? payload.symbol.trim().toUpperCase() : '';
  const rawInterval = typeof payload.interval === 'string' ? payload.interval.trim() : '';

  if (!rawSymbol || !/^[A-Z0-9]{5,20}$/.test(rawSymbol)) {
    return { error: 'Invalid symbol. Use uppercase market symbols like BTCUSDT.' };
  }

  if (!ALLOWED_INTERVALS.has(rawInterval)) {
    return { error: `Invalid interval. Allowed values: ${Array.from(ALLOWED_INTERVALS).join(', ')}` };
  }

  return { symbol: rawSymbol, interval: rawInterval };
}

function buildDirectionalTilt(closes) {
  if (closes.length < MA_PERIOD) {
    return { bias: 'NEUTRAL', tilt_strength: 0 };
  }

  const ma = closes.slice(-MA_PERIOD).reduce((sum, c) => sum + c, 0) / MA_PERIOD;
  const latest = closes[closes.length - 1];
  if (!ma || !latest) {
    return { bias: 'NEUTRAL', tilt_strength: 0 };
  }

  const deviation = (latest - ma) / ma;
  const tiltStrength = clamp(Math.abs(deviation) / 0.02, 0, 1);

  if (Math.abs(deviation) < 0.001) {
    return { bias: 'NEUTRAL', tilt_strength: tiltStrength };
  }

  return {
    bias: deviation > 0 ? 'UP' : 'DOWN',
    tilt_strength: tiltStrength
  };
}

function computeStressMetrics(klines) {
  const closes = [];
  const volumes = [];

  for (const candle of klines) {
    if (!Array.isArray(candle) || candle.length < 6) continue;
    const close = safeNumber(candle[4]);
    const volume = safeNumber(candle[5]);
    if (close === null || volume === null || close <= 0 || volume < 0) continue;
    closes.push(close);
    volumes.push(volume);
  }

  if (closes.length < 30 || volumes.length < 30) {
    return { error: 'Insufficient market data returned from Binance.' };
  }

  const absReturns = [];
  for (let i = 1; i < closes.length; i += 1) {
    absReturns.push(Math.abs(Math.log(closes[i] / closes[i - 1])));
  }

  const returnVol = stdDev(absReturns);
  const normalizedReturnVol = clamp(returnVol / 0.01, 0, 1);

  const volumeLookback = volumes.slice(-30);
  const avgVolume = volumeLookback.reduce((sum, v) => sum + v, 0) / volumeLookback.length;
  const latestVolume = volumes[volumes.length - 1];
  const volumeSpikeRatio = avgVolume > 0 ? latestVolume / avgVolume : 1;
  const normalizedVolumeSpike = clamp((volumeSpikeRatio - 1) / 2, 0, 1);

  const stressScore = clamp((normalizedReturnVol * 0.7) + (normalizedVolumeSpike * 0.3), 0, 1);
  const dataQuality = clamp(closes.length / MAX_LIMIT, 0, 1);
  const confidence = clamp(0.6 + (Math.abs(stressScore - 0.5) * 0.6) + (dataQuality * 0.2), 0, 1);

  return {
    stress_score: Number(stressScore.toFixed(4)),
    stress_level: classifyStress(stressScore),
    confidence: Number(confidence.toFixed(4)),
    directional_tilt: buildDirectionalTilt(closes)
  };
}

async function fetchBinanceKlines(symbol, interval) {
  const url = new URL(BINANCE_KLINES_URL);
  url.searchParams.set('symbol', symbol);
  url.searchParams.set('interval', interval);
  url.searchParams.set('limit', String(MAX_LIMIT));

  const response = await fetch(url.toString(), {
    method: 'GET',
    headers: { accept: 'application/json' }
  });

  if (!response.ok) {
    throw new Error(`Binance request failed with status ${response.status}`);
  }

  return response.json();
}

export default {
  async fetch(request) {
    if (request.method === 'OPTIONS') {
      return corsPreflightResponse();
    }

    if (request.method !== 'POST') {
      return jsonResponse({ error: 'Method not allowed. Use POST.' }, 405);
    }

    let payload;
    try {
      payload = await request.json();
    } catch {
      return jsonResponse({ error: 'Invalid JSON body.' }, 400);
    }

    const validated = parseAndValidatePayload(payload);
    if (validated.error) {
      return jsonResponse({ error: validated.error }, 400);
    }

    try {
      const klines = await fetchBinanceKlines(validated.symbol, validated.interval);
      if (!Array.isArray(klines)) {
        return jsonResponse({ error: 'Unexpected Binance response format.' }, 502);
      }

      const metrics = computeStressMetrics(klines);
      if (metrics.error) {
        return jsonResponse({ error: metrics.error }, 502);
      }

      return jsonResponse({
        symbol: validated.symbol,
        ...metrics
      });
    } catch {
      return jsonResponse({
        error: 'Failed to fetch market data from Binance.'
      }, 502);
    }
  }
};
