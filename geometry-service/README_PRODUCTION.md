# Production Deployment Guide

## New Features (v11.0.0)

### 1. Circuit Breaker
- **Purpose**: Prevents cascade failures
- **Threshold**: Opens after 5 consecutive failures
- **Timeout**: 60 seconds before testing recovery
- **Endpoint**: `GET /circuit-breaker` to check state
- **Manual Reset**: `POST /circuit-breaker/reset`

### 2. Graceful Degradation
- **Tier 1** (0.95 confidence): Full B-Rep + AAGNet
- **Tier 2** (0.75 confidence): Mesh-based heuristics
- **Tier 3** (0.60 confidence): Point cloud basic
- **Tier 4** (0.40 confidence): Mesh visualization only

### 3. Retry Logic
- Exponential backoff: 1s, 2s, 4s, 8s
- Max 3 attempts for transient errors
- No retry for permanent/systemic errors

### 4. Dead Letter Queue
- Stores all failures with context
- Accessible via `/dlq/stats` and `/dlq/failures`
- Error classification: transient, permanent, systemic

### 5. Monitoring
- **Endpoint**: `GET /metrics`
- Tracks circuit breaker state, DLQ statistics
- Includes processing tier distribution

## Installation
```bash
cd geometry-service/

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Initialize AAGNet (if not already done)
git clone https://github.com/whjdark/AAGNet.git AAGNet-main
# Download weights to AAGNet-main/weights/

# Run migrations
# (Execute the SQL migration in Supabase dashboard)
```

## Running Smoke Tests
```bash
# Start service locally
python app.py

# In another terminal
python tests/smoke_tests.py

# Or against deployed service
GEOMETRY_SERVICE_URL=https://your-render-service.onrender.com python tests/smoke_tests.py
```

## Expected Smoke Test Output
```
Running smoke tests against: http://localhost:5000
============================================================
TEST: Service Health Check
============================================================
  Circuit breaker state: CLOSED
✅ PASS (0.05s)

...

============================================================
SMOKE TEST SUMMARY
============================================================
Total tests: 7
Passed: 7
Failed: 0
Total time: 3.45s
```

## Monitoring in Production

### Check Circuit Breaker State
```bash
curl https://your-service.onrender.com/metrics
```

### View Recent Failures
```bash
curl https://your-service.onrender.com/dlq/failures?limit=10
```

### Manual Circuit Breaker Reset (if needed)
```bash
curl -X POST https://your-service.onrender.com/circuit-breaker/reset
```

## Troubleshooting

### Circuit Breaker Open
**Symptom**: All requests fail with "Circuit breaker is OPEN"
**Cause**: 5+ consecutive failures detected
**Fix**: 
1. Check `/dlq/failures` to identify root cause
2. Wait 60s for automatic recovery test
3. Or manually reset: `POST /circuit-breaker/reset`

### High Tier 4 Usage
**Symptom**: Most requests return `processing_tier: "tier_4_basic"`
**Cause**: AAGNet consistently failing
**Fix**:
1. Check AAGNet model loading: `GET /health` → `aagnet_loaded: false`
2. Verify model weights exist in `AAGNet-main/weights/`
3. Check memory constraints (512MB may be insufficient)

### Dead Letter Queue Growing
**Symptom**: `total_failures_24h` increasing rapidly
**Cause**: Persistent errors with uploaded files
**Fix**:
1. Review error patterns: `GET /dlq/failures?error_type=permanent`
2. Identify common file characteristics
3. Improve preprocessing/validation

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Tier 1 Latency | < 10s | Typical parts |
| Tier 1 Latency | < 30s | Complex assemblies |
| Circuit Breaker Failures | < 5 per hour | Healthy system |
| Tier 4 Fallback Rate | < 5% | Most requests should complete Tier 1-2 |
| DLQ Growth | < 10 per day | Indicates preprocessing quality |

## Render Deployment Configuration

### Environment Variables
```
SUPABASE_URL=your-url
SUPABASE_SERVICE_ROLE_KEY=your-key
DEBUG=false
```

### Health Check
- **Path**: `/health`
- **Expected**: `200 OK`
- **Timeout**: 30s

### Scaling Recommendations
- **Current**: 1 instance (512MB) - Single point of failure
- **Recommended**: 2 instances (1GB each) for 99.9% SLA
- **Load Balancer**: Configure sticky sessions

## Database Schema

### New Table: `failed_cad_analyses`
```sql
id               UUID PRIMARY KEY
correlation_id   TEXT NOT NULL
file_path        TEXT NOT NULL
error_type       TEXT CHECK (IN 'transient', 'permanent', 'systemic')
error_message    TEXT NOT NULL
error_details    JSONB
retry_count      INTEGER
traceback        TEXT
created_at       TIMESTAMP WITH TIME ZONE
```

## Next Steps

1. **Deploy to Staging**: Test smoke tests against staging environment
2. **Load Testing**: Verify circuit breaker under high load
3. **Monitoring Dashboard**: Create Grafana/equivalent dashboard for metrics
4. **Alerting**: Set up alerts for circuit breaker open, high DLQ growth
5. **Upgrade to 2 Instances**: Eliminate single point of failure
```

---

## FILE LOCATIONS SUMMARY
```
geometry-service/
├── app.py                          ← MODIFIED (add circuit breaker, DLQ, graceful degradation)
├── circuit_breaker.py              ← NEW
├── retry_utils.py                  ← NEW
├── dead_letter_queue.py            ← NEW
├── graceful_degradation.py         ← NEW
├── requirements.txt                ← UPDATED (add requests)
├── README_PRODUCTION.md            ← NEW
└── tests/
    ├── smoke_tests.py              ← NEW
    └── test_files/
        └── (place test STEP files here)

supabase/migrations/
└── 20250107_add_failed_analyses_table.sql  ← NEW
