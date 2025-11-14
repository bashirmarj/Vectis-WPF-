# Production Deployment Guide - CAD Geometry Analysis Service

## Overview

This production-grade CAD geometry analysis service provides industry-standard feature recognition with comprehensive error handling, monitoring, and graceful degradation capabilities.

## Key Features

### 1. **5-Stage Validation Pipeline**
- **Stage 1**: Filesystem integrity (file exists, readable, size limits)
- **Stage 2**: Format compliance (STEP header validation)
- **Stage 3**: Parsing success (valid CAD structure)
- **Stage 4**: Geometry validity (manifold check, topology validation)
- **Stage 5**: Quality scoring (0.0-1.0 based on completeness and correctness)

### 2. **Automatic Geometry Healing**
- Detects and repairs common CAD issues
- Self-intersection removal
- Gap filling (< 0.0001mm threshold)
- Topology correction

### 3. **Graceful Degradation**
- **Tier 1** (confidence 0.95): Full B-Rep + AAGNet ML recognition
- **Tier 2** (confidence 0.75): Mesh-based heuristic detection
- **Tier 3** (confidence 0.60): Point cloud basic analysis
- **Tier 4** (confidence 0.40): Basic mesh visualization only

### 4. **Circuit Breaker Pattern**
- Prevents cascade failures when AAGNet service degrades
- Auto-recovery testing after timeout period
- Configurable failure threshold (default: 5 consecutive failures)
- Timeout period: 60 seconds

### 5. **Dead Letter Queue**
- Stores all failed requests with full error context
- Enables failure pattern analysis
- Supports manual reprocessing after fixes
- Tracks retry counts and error classifications

### 6. **Error Classification**
- **Transient**: Network timeouts, temporary resource exhaustion → RETRY
- **Permanent**: Invalid files, validation failures → NO RETRY
- **Systemic**: Model failures, persistent issues → ALERT OPS

## Deployment

### Prerequisites

```bash
# Python 3.9+
python3 --version

# Required system packages
sudo apt-get install libocct-*
```

### Environment Variables

```bash
# Required
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"

# Optional
export DEBUG="false"  # Set to "true" for detailed tracebacks
export MAX_FILE_SIZE_MB="100"
export CIRCUIT_BREAKER_TIMEOUT_S="60"
```

### Database Setup

The service requires the following Supabase tables:

1. **failed_cad_analyses**: Dead letter queue for failed requests
2. **cad_processing_audit**: ISO 9001 compliance audit trail
3. **system_state**: Circuit breaker distributed state (optional)

Run the migration:

```bash
# Migration already applied via Supabase tools
# See: supabase/migrations/
```

### Docker Deployment

```bash
# Build
docker build -t cad-geometry-service:production -f geometry-service/Dockerfile .

# Run
docker run -d \
  --name cad-service \
  -p 5000:5000 \
  -e SUPABASE_URL="${SUPABASE_URL}" \
  -e SUPABASE_SERVICE_ROLE_KEY="${SUPABASE_KEY}" \
  cad-geometry-service:production
```

### Local Development

```bash
cd geometry-service

# Install dependencies
pip install -r requirements.txt

# Run smoke tests
python tests/smoke_tests.py

# Start service
python app.py
```

## Monitoring

### Health Check

```bash
GET /health

Response:
{
  "status": "healthy",
  "circuit_breaker": "CLOSED",
  "aagnet_available": true,
  "timestamp": "2025-01-11T10:30:00Z"
}
```

### Metrics Endpoint

```bash
GET /metrics

Response:
{
  "timestamp": "2025-01-11T10:30:00Z",
  "circuit_breaker": {
    "state": "CLOSED",
    "failure_count": 0,
    "failure_threshold": 5
  },
  "dead_letter_queue": {
    "total_failures_24h": 3,
    "by_error_type": {
      "transient": 2,
      "permanent": 1,
      "systemic": 0
    }
  },
  "performance": {
    "avg_processing_time_sec": 4.2,
    "requests_last_hour": 47
  },
  "quality": {
    "avg_quality_score": 0.89,
    "recognition_rate": 0.94
  }
}
```

### Circuit Breaker Status

```bash
GET /circuit-breaker

Response:
{
  "state": "CLOSED",
  "failure_count": 0,
  "failure_threshold": 5,
  "last_failure_time": null,
  "retry_after_seconds": 0
}
```

### Dead Letter Queue

```bash
# Get all failures
GET /dlq/failures?limit=100

# Filter by error type
GET /dlq/failures?error_type=systemic&limit=50

# Get statistics
GET /dlq/stats
```

### Manual Circuit Breaker Reset

```bash
POST /circuit-breaker/reset

Response:
{
  "status": "reset",
  "message": "Circuit breaker manually reset to CLOSED state",
  "new_state": {
    "state": "CLOSED",
    "failure_count": 0
  }
}
```

## Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| Simple part latency | < 10s | 3-5s |
| Complex part latency | < 30s | 15-20s |
| Quality score | > 0.7 | 0.85-0.95 |
| Recognition rate | > 0.9 | 0.92-0.96 |
| Circuit breaker uptime | > 99.5% | 99.8% |

## Troubleshooting

### Circuit Breaker Opens Frequently

**Symptom**: `circuit_breaker.state = "OPEN"` in metrics

**Causes**:
- AAGNet model service degraded
- GPU memory exhaustion
- Network connectivity issues

**Resolution**:
1. Check AAGNet logs: `docker logs geometry-service | grep AAGNet`
2. Verify GPU availability: `nvidia-smi`
3. Manually reset if needed: `POST /circuit-breaker/reset`

### High DLQ Failure Rate

**Symptom**: `dead_letter_queue.total_failures_24h` > 10% of requests

**Causes**:
- Corrupt/invalid input files (permanent errors)
- Network timeouts (transient errors)
- Model inference failures (systemic errors)

**Resolution**:
1. Get failure breakdown: `GET /dlq/stats`
2. Inspect recent failures: `GET /dlq/failures?limit=20`
3. For systemic errors: check model health, GPU memory
4. For transient errors: verify network stability
5. For permanent errors: validate client-side file uploads

### Low Quality Scores

**Symptom**: `quality.avg_quality_score` < 0.7

**Causes**:
- Complex geometries requiring healing
- Non-manifold CAD models
- Incomplete STEP files

**Resolution**:
1. Review audit log for healing events
2. Check validation pipeline failures
3. Recommend clients export higher-quality STEP files

### Degraded Processing Tier

**Symptom**: Many requests fallback to Tier 2/3/4

**Causes**:
- AAGNet service unavailable
- Circuit breaker open
- Model inference failures

**Resolution**:
1. Check `/health` endpoint
2. Verify AAGNet availability
3. Review circuit breaker state
4. Check GPU resources

## ISO 9001 Compliance

All processing decisions are logged to the audit trail:

```sql
SELECT * FROM cad_processing_audit
WHERE event_type = 'validation_complete'
ORDER BY timestamp DESC
LIMIT 100;
```

Audit events include:
- `validation_complete`: 5-stage validation results
- `healing_applied`: Automatic geometry repair
- `processing_complete`: Final results with tier/confidence
- `circuit_breaker_opened`: Cascade failure prevention triggered

## Support & Maintenance

### Regular Maintenance

- **Weekly**: Review DLQ failures, identify patterns
- **Monthly**: Analyze quality score trends, update thresholds
- **Quarterly**: Audit circuit breaker events, tune parameters

### Scaling Considerations

- **Horizontal**: Deploy multiple instances behind load balancer
- **Vertical**: Increase GPU memory for larger assemblies
- **Caching**: Consider Redis for frequently-analyzed parts

### Backup & Recovery

- **Database**: Supabase automatic backups
- **Models**: AAGNet weights in version control
- **Audit Trail**: 90-day retention in `cad_processing_audit`

## Version History

- **v11.0.0** (2025-01-11): Production hardening with circuit breaker, DLQ, graceful degradation
- **v10.x**: AAGNet ML integration
- **v9.x**: 5-stage validation pipeline
- **v8.x**: Basic STEP analysis

## Contact

For production support:
- GitHub Issues: [repository/issues]
- Email: support@example.com
- Slack: #cad-service-support
