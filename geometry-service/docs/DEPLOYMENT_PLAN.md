# VECTIS MACHINING - PRODUCTION DEPLOYMENT PLAN
**Version 2.0.0 - BRepNet + Geometric Fallback Architecture**  
**Target: 50-100 STEP files/day on CPU-based infrastructure**

---

## EXECUTIVE SUMMARY

**What Changed:**
- ❌ Removed: 15,599 lines of overlapping code (29 Python modules)
- ✅ Replaced with: 800 lines core + 400 lines BRepNet wrapper + 300 lines geometric fallback
- ✅ Platform: DigitalOcean App Platform (NOT Render.com - no GPU support)
- ✅ Model: BRepNet with pre-trained weights (89.96% accuracy)
- ✅ Cost: $115/month infrastructure (vs $170+ for GPU)

**Key Improvements:**
- Single responsibility modules (no overlap)
- ONNX-optimized CPU inference (2-5 seconds per file)
- Face-level feature mapping for 3D highlighting
- Geometric fallback for turning features
- Production error handling and logging

---

## DEPLOYMENT TIMELINE (8 WEEKS)

### Week 1-2: Local Development & Testing
**Goal:** Verify BRepNet integration works on sample files

#### Tasks:
1. **Set up local environment**
   ```bash
   # Clone fresh repository
   git clone https://github.com/bashirmarj/vectismachining.git
   cd vectismachining
   
   # Create clean geometry-service-v2 directory
   mkdir geometry-service-v2
   cd geometry-service-v2
   
   # Copy new files
   cp /path/to/app.py .
   cp /path/to/brepnet_wrapper.py .
   cp /path/to/geometric_fallback.py .
   cp /path/to/requirements.txt .
   cp /path/to/Dockerfile .
   ```

2. **Install dependencies**
   ```bash
   # Install miniconda
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   
   # Create environment
   conda create -n vectis python=3.10
   conda activate vectis
   
   # Install pythonocc-core
   conda install -c conda-forge pythonocc-core=7.7.0
   
   # Install Python requirements
   pip install -r requirements.txt
   ```

3. **Download BRepNet model**
   ```bash
   mkdir models
   cd models
   
   # Clone BRepNet repository
   git clone https://github.com/AutodeskAILab/BRepNet.git
   
   # Copy pre-trained checkpoint
   cp BRepNet/example_files/pretrained_models/pretrained_s2.0.0_step_all_features_0519_073100.ckpt \
      brepnet_pretrained.ckpt
   
   # Convert to ONNX (run Python script)
   python ../convert_to_onnx.py
   ```

4. **Test with sample STEP files**
   ```bash
   # Start development server
   python app.py
   
   # Test endpoint
   curl -X POST http://localhost:8080/analyze \
     -F "file=@test_files/simple_block.step"
   ```

**Acceptance Criteria:**
- ✅ Service starts without errors
- ✅ Successfully processes sample STEP file
- ✅ Returns mesh data with face mapping
- ✅ BRepNet detects at least 1 feature
- ✅ Processing time < 10 seconds

---

### Week 3-4: Backend API & Integration

#### Tasks:
1. **Update Supabase Edge Function** (`analyze-cad`)
   - Replace Python service calls with new `/analyze` endpoint
   - Add correlation ID tracking
   - Implement timeout handling (30 seconds max)

2. **Frontend Integration**
   - Update `CADViewer.tsx` to use new mesh format
   - Implement face highlighting from `face_mapping`
   - Add feature tree display for recognized features

3. **Database Schema Updates**
   ```sql
   -- Add new columns to part_features table
   ALTER TABLE part_features ADD COLUMN IF NOT EXISTS
     model_version VARCHAR(50) DEFAULT 'brepnet-2.0';
   
   ALTER TABLE part_features ADD COLUMN IF NOT EXISTS
     confidence_score FLOAT DEFAULT 0.0;
   
   ALTER TABLE part_features ADD COLUMN IF NOT EXISTS
     recognition_method VARCHAR(50) DEFAULT 'ml';  -- 'ml' or 'geometric'
   ```

4. **Testing Suite**
   - Create test suite with 20 diverse STEP files
   - Prismatic parts (blocks, brackets): 10 files
   - Turning parts (shafts, pulleys): 5 files
   - Mixed geometry: 5 files

**Acceptance Criteria:**
- ✅ Edge function successfully calls new service
- ✅ Frontend displays features correctly
- ✅ Database stores confidence scores
- ✅ 18/20 test files process successfully (90%)

---

### Week 5-6: Docker & DigitalOcean Deployment

#### Tasks:
1. **Build Docker Image**
   ```bash
   # Build locally first
   docker build -t vectis-geometry-service:2.0.0 .
   
   # Test container
   docker run -p 8080:8080 \
     -e SUPABASE_URL=$SUPABASE_URL \
     -e SUPABASE_SERVICE_ROLE_KEY=$SUPABASE_KEY \
     vectis-geometry-service:2.0.0
   ```

2. **Set up DigitalOcean App Platform**
   ```yaml
   # app.yaml
   name: vectis-geometry-service
   region: nyc
   
   services:
     - name: geometry-api
       dockerfile_path: Dockerfile
       source_dir: geometry-service-v2
       instance_count: 1
       instance_size_slug: apps-d-4vcpu-16gb  # $84/month
       http_port: 8080
       
       health_check:
         http_path: /health
         initial_delay_seconds: 60
       
       envs:
         - key: SUPABASE_URL
           scope: RUN_TIME
           type: SECRET
         - key: SUPABASE_SERVICE_ROLE_KEY
           scope: RUN_TIME
           type: SECRET
   ```

3. **Deploy to DigitalOcean**
   ```bash
   # Install doctl CLI
   brew install doctl  # or apt-get install doctl
   
   # Authenticate
   doctl auth init
   
   # Create app from spec
   doctl apps create --spec app.yaml
   
   # Monitor deployment
   doctl apps list
   doctl apps logs <app-id> --follow
   ```

4. **Configure DNS**
   - Add CNAME record: `geometry-api.vectismachining.com` → DigitalOcean app URL
   - Update Supabase Edge Function URLs

**Acceptance Criteria:**
- ✅ Docker image builds successfully
- ✅ Service deploys to DigitalOcean
- ✅ Health check returns 200 OK
- ✅ Can process STEP file from Supabase Edge Function
- ✅ Response time < 5 seconds for medium files

---

### Week 7-8: Production Hardening & Monitoring

#### Tasks:
1. **Implement Caching**
   ```python
   # Add Redis for file hash caching
   import redis
   
   cache = redis.Redis(host='redis-cache', port=6379)
   
   def check_cache(file_hash):
       cached = cache.get(f"analysis:{file_hash}")
       if cached:
           return json.loads(cached)
       return None
   
   def store_to_cache(file_hash, result):
       cache.setex(f"analysis:{file_hash}", 86400, json.dumps(result))
   ```

2. **Add Monitoring**
   - Set up Sentry for error tracking
   - Add Prometheus metrics endpoint
   - Configure DigitalOcean alerts:
     - CPU > 80% for 5 minutes
     - Memory > 90%
     - Error rate > 5%

3. **Performance Optimization**
   - Enable ONNX graph optimization
   - Implement request queuing for burst traffic
   - Add connection pooling for Supabase

4. **Load Testing**
   ```bash
   # Use k6 or locust for load testing
   k6 run --vus 10 --duration 60s load_test.js
   
   # Target metrics:
   # - 95th percentile < 10 seconds
   # - Throughput: 100 requests/hour
   # - Error rate < 1%
   ```

**Acceptance Criteria:**
- ✅ Cache hit rate > 30%
- ✅ Error tracking configured
- ✅ Alerts fire correctly
- ✅ Load test passes (10 concurrent users)
- ✅ Zero downtime during normal operations

---

## INFRASTRUCTURE ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRODUCTION STACK                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CLIENT (React + Three.js)                                       │
│         │                                                        │
│         ▼                                                        │
│  SUPABASE EDGE FUNCTION (analyze-cad)                           │
│         │                                                        │
│         ├──────────────────────┐                                │
│         │                      │                                │
│         ▼                      ▼                                │
│  DigitalOcean App Platform   Redis Cache                        │
│  ┌──────────────────────┐   (Managed)                          │
│  │ Geometry Service     │   $15/month                           │
│  │ ├─ app.py           │                                        │
│  │ ├─ brepnet_wrapper  │                                        │
│  │ ├─ geometric_fallback│                                       │
│  │ └─ models/          │                                        │
│  │    └─ brepnet.onnx  │                                        │
│  └──────────────────────┘                                        │
│  4 vCPU, 16GB RAM                                                │
│  $84/month                                                       │
│         │                                                        │
│         ▼                                                        │
│  Supabase PostgreSQL (store results)                            │
│  ┌──────────────────────────────────────┐                       │
│  │ - quotation_submissions             │                       │
│  │ - cad_meshes                         │                       │
│  │ - part_features                      │                       │
│  └──────────────────────────────────────┘                       │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│ MONITORING & OBSERVABILITY                                       │
│ - Sentry (error tracking)                                       │
│ - DigitalOcean Insights (metrics)                               │
│ - Supabase Dashboard (database monitoring)                      │
└─────────────────────────────────────────────────────────────────┘
```

**Monthly Costs:**
- DigitalOcean App Platform: $84
- Redis Cache: $15
- DigitalOcean Bandwidth: ~$5
- Sentry (Developer): $0 (free tier)
- **Total: ~$104/month**

---

## SCALING STRATEGY

### Current Capacity (CPU-only)
- **Throughput:** 10-20 files/hour (4 vCPU)
- **Target:** 50-100 files/day = 2-4 files/hour
- **Headroom:** 3-5x capacity buffer ✅

### If Scaling Needed (>100 files/day)

#### Option A: Horizontal Scaling (CPU)
```yaml
# Increase instance count
instance_count: 2  # $168/month

# Add load balancer automatically handled by DO App Platform
```

#### Option B: GPU Acceleration (>500 files/day)
```yaml
# Switch to RunPod Serverless
# Keep API on DigitalOcean, offload inference to GPU

services:
  - name: geometry-api
    instance_size: apps-s-1vcpu-2gb  # $12/month (API only)
  
# Add RunPod webhook endpoint
# Cost: $0.34/hr × 1hr/day = $10/month for 100 files/day
```

---

## ROLLBACK PLAN

If production deployment fails:

1. **Immediate Rollback**
   ```bash
   # Revert to previous deployment
   doctl apps create-deployment <app-id> \
     --wait --image=<previous-image-tag>
   ```

2. **Keep old service running**
   - Don't delete old `geometry-service` until new one proven
   - Switch Edge Function URL back to old endpoint

3. **Database Rollback**
   ```sql
   -- Remove new columns if needed
   ALTER TABLE part_features DROP COLUMN IF EXISTS model_version;
   ALTER TABLE part_features DROP COLUMN IF EXISTS confidence_score;
   ```

---

## SUCCESS METRICS

### Week 2 (Development)
- [ ] BRepNet processes 10 test files successfully
- [ ] Average processing time < 10 seconds
- [ ] Zero crashes on valid STEP files

### Week 4 (Integration)
- [ ] Edge Function successfully calls new service
- [ ] Frontend displays features correctly
- [ ] 90% test file success rate

### Week 6 (Deployment)
- [ ] Service deployed to DigitalOcean
- [ ] 99% uptime over 1 week
- [ ] Average response time < 5 seconds

### Week 8 (Production)
- [ ] Processing 50+ files/day
- [ ] Cache hit rate > 30%
- [ ] Error rate < 2%
- [ ] Customer satisfaction with feature accuracy

---

## MAINTENANCE & SUPPORT

### Daily Monitoring
- Check DigitalOcean dashboard for alerts
- Review Sentry for new errors
- Monitor cache hit rates

### Weekly Tasks
- Review slow query logs
- Check disk usage (STEP file storage)
- Analyze feature recognition accuracy reports

### Monthly Tasks
- Update dependencies (security patches)
- Review and optimize costs
- Plan model improvements based on user feedback

---

## CONTACTS & ESCALATION

**Technical Lead:** Bashir Marj  
**Deployment Support:** DigitalOcean Support (Business tier)  
**Model Issues:** BRepNet GitHub Issues  
**Infrastructure:** Supabase Support  

**Escalation Path:**
1. Check logs (DigitalOcean + Sentry)
2. Restart service if needed
3. Rollback if critical failure
4. Contact support if infrastructure issue

---

## NEXT STEPS (Post-Deployment)

### 3-Month Roadmap
1. **Month 1:** Stabilize production, gather accuracy metrics
2. **Month 2:** Fine-tune confidence thresholds based on real data
3. **Month 3:** Begin AAGNet training on custom dataset

### 6-Month Roadmap
1. Add geometric parameter extraction (hole diameters, pocket dimensions)
2. Implement advanced turning feature detection (threads, tapers)
3. Train AAGNet on mixed prismatic+turning dataset

### 12-Month Roadmap
1. Achieve 95%+ automation rate (50% → 95%)
2. Deploy GPU acceleration for sub-second response times
3. Expand to 500+ files/day capacity

---

**Document Version:** 2.0.0  
**Last Updated:** November 13, 2025  
**Next Review:** Post-deployment (Week 8)
