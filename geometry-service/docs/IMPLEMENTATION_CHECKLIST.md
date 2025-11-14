# IMPLEMENTATION CHECKLIST - Vectis Machining v2.0
**Complete Task List for Clean BRepNet Deployment**

---

## PRE-IMPLEMENTATION (Day 0)

### Planning & Coordination
- [ ] Review complete deployment plan with team
- [ ] Identify deployment window (recommended: weekend/off-hours)
- [ ] Notify stakeholders of upcoming changes
- [ ] Set up rollback procedures
- [ ] Back up production database

### Environment Preparation
- [ ] Create DigitalOcean account (if needed)
- [ ] Set up Sentry error tracking account
- [ ] Verify Supabase production credentials
- [ ] Create GitHub branch: `refactor/v2.0-clean-architecture`
- [ ] Tag legacy code: `v1.0.0-legacy`

---

## WEEK 1: REPOSITORY CLEANUP

### Day 1-2: Code Removal
- [ ] Archive old geometry-service to `archive/legacy-geometry-service/`
- [ ] Delete 9 duplicate recognizer files (6,269 lines)
- [ ] Delete 5 unused ML implementation files (2,664 lines)
- [ ] Delete 6 infrastructure overhead files (1,889 lines)
- [ ] Remove duplicate frontend components (5 files)
- [ ] Commit with message: "refactor: Remove 12,900 lines of overlapping code"

### Day 3: New Structure Creation
- [ ] Create `geometry-service-v2/` directory
- [ ] Copy new `app.py` (800 lines)
- [ ] Copy new `brepnet_wrapper.py` (400 lines)
- [ ] Copy new `geometric_fallback.py` (300 lines)
- [ ] Copy essential utilities from v1:
  - [ ] `machining_estimator.py`
  - [ ] `routing_selector_industrial.py`
  - [ ] `util.py`
- [ ] Copy new `requirements.txt`
- [ ] Copy new `Dockerfile`
- [ ] Create `models/` directory

### Day 4: Documentation
- [ ] Copy `DEPLOYMENT_PLAN.md` to repository
- [ ] Copy `CLEANUP_PLAN.md` to repository
- [ ] Copy `QUICKSTART.md` to repository
- [ ] Update main `README.md` with v2.0 information
- [ ] Create `MIGRATION_GUIDE.md` for upgrading from v1.0

### Day 5: Git Cleanup
- [ ] Run `git gc --prune=now --aggressive`
- [ ] Push branch to GitHub
- [ ] Create pull request
- [ ] Request code review from team

**Week 1 Success Criteria:**
- ✅ Repository reduced by 12,900+ lines
- ✅ Clean v2.0 structure in place
- ✅ All documentation complete
- ✅ PR created and under review

---

## WEEK 2: LOCAL DEVELOPMENT

### Day 1: Environment Setup
- [ ] Install Miniconda
- [ ] Create conda environment: `vectis`
- [ ] Install pythonocc-core 7.7.0
- [ ] Install Python dependencies from requirements.txt
- [ ] Set up `.env` file with Supabase credentials

### Day 2: BRepNet Model Acquisition
- [ ] Clone BRepNet repository
- [ ] Locate pre-trained checkpoint
- [ ] Download model: `pretrained_s2.0.0_step_all_features_0519_073100.ckpt`
- [ ] Place in `models/` directory
- [ ] Verify model loads without errors

### Day 3-4: Local Testing
- [ ] Start service: `python app.py`
- [ ] Test health endpoint: `curl http://localhost:8080/health`
- [ ] Test with simple STEP file (100-500 faces)
- [ ] Test with medium STEP file (500-2000 faces)
- [ ] Test with complex STEP file (2000+ faces)
- [ ] Verify tessellation works correctly
- [ ] Verify BRepNet recognizes features
- [ ] Check processing times (< 10 seconds target)

### Day 5: Bug Fixes
- [ ] Address any errors found during testing
- [ ] Optimize tessellation parameters if needed
- [ ] Adjust confidence thresholds
- [ ] Document any issues in GitHub issues

**Week 2 Success Criteria:**
- ✅ Service runs without errors
- ✅ Processes 10 diverse test files successfully
- ✅ Average processing time < 10 seconds
- ✅ BRepNet detects features with >70% confidence

---

## WEEK 3-4: BACKEND INTEGRATION

### Week 3, Day 1-2: Database Updates
- [ ] Create database migration for new columns
- [ ] Add `model_version VARCHAR(50)` to `part_features`
- [ ] Add `confidence_score FLOAT` to `part_features`
- [ ] Add `recognition_method VARCHAR(50)` to `part_features`
- [ ] Test migration on local Supabase
- [ ] Apply migration to production (during maintenance window)

### Week 3, Day 3-4: Supabase Edge Function Update
- [ ] Update `analyze-cad/index.ts` to use new `/analyze` endpoint
- [ ] Add correlation ID generation
- [ ] Update mesh data storage to handle new format
- [ ] Update feature storage to include confidence scores
- [ ] Add error handling for new service
- [ ] Test Edge Function locally with Supabase CLI

### Week 3, Day 5: Integration Testing
- [ ] Upload test STEP file via frontend
- [ ] Verify Edge Function calls geometry service
- [ ] Check mesh data stored correctly in `cad_meshes`
- [ ] Check features stored correctly in `part_features`
- [ ] Verify correlation IDs appear in logs
- [ ] Test error scenarios (invalid file, timeout, etc.)

### Week 4, Day 1-3: Frontend Updates
- [ ] Update `CADViewer.tsx` to use new mesh format
- [ ] Implement face highlighting using `face_mapping`
- [ ] Update `FeatureTree.tsx` to show confidence scores
- [ ] Add "Model: BRepNet v2.0" indicator
- [ ] Test 3D visualization with new mesh data
- [ ] Verify feature highlighting works

### Week 4, Day 4-5: End-to-End Testing
- [ ] Upload 20 diverse STEP files through full stack
- [ ] Verify all components work together
- [ ] Check database for correct data storage
- [ ] Test on different browsers (Chrome, Firefox, Safari)
- [ ] Test on mobile devices
- [ ] Document any issues

**Week 3-4 Success Criteria:**
- ✅ Database migrations applied successfully
- ✅ Edge Function calls new service correctly
- ✅ Frontend displays features with highlighting
- ✅ 18/20 test files process successfully (90%)
- ✅ No crashes or data corruption

---

## WEEK 5-6: DOCKER & DEPLOYMENT

### Week 5, Day 1-2: Docker Build
- [ ] Build Docker image locally
- [ ] Test container runs without errors
- [ ] Test health endpoint in container
- [ ] Test analyze endpoint in container
- [ ] Verify model loads correctly in container
- [ ] Check memory usage (should be < 2GB)
- [ ] Optimize Dockerfile if needed

### Week 5, Day 3: Container Registry
- [ ] Tag image: `vectis-geometry-service:2.0.0`
- [ ] Push to Docker Hub or DigitalOcean Container Registry
- [ ] Verify image pulls correctly
- [ ] Test image on clean machine

### Week 5, Day 4-5: DigitalOcean Setup
- [ ] Create DigitalOcean App Platform app
- [ ] Configure from `app.yaml` spec
- [ ] Set instance size: `apps-d-4vcpu-16gb`
- [ ] Add environment variables (Supabase credentials)
- [ ] Configure health check endpoint
- [ ] Set up custom domain: `geometry-api.vectismachining.com`

### Week 6, Day 1-2: Staging Deployment
- [ ] Deploy to staging environment
- [ ] Run smoke tests
- [ ] Monitor logs for 24 hours
- [ ] Check CPU/memory usage
- [ ] Load test with 50 files
- [ ] Verify no memory leaks

### Week 6, Day 3: Production Deployment
- [ ] Schedule deployment window (off-hours)
- [ ] Announce downtime to users
- [ ] Deploy to production
- [ ] Run health checks
- [ ] Test with production data
- [ ] Monitor for 1 hour
- [ ] Verify Edge Function connects correctly

### Week 6, Day 4-5: Post-Deployment Monitoring
- [ ] Monitor error rates (target < 2%)
- [ ] Check response times (target < 5 seconds)
- [ ] Monitor CPU usage (should be < 80%)
- [ ] Check memory usage (should be < 14GB)
- [ ] Review Sentry errors
- [ ] Address any critical issues

**Week 5-6 Success Criteria:**
- ✅ Docker image builds and runs correctly
- ✅ Service deployed to DigitalOcean
- ✅ Health check returns 200 OK
- ✅ Processes production files successfully
- ✅ Average response time < 5 seconds
- ✅ 99%+ uptime over 1 week

---

## WEEK 7-8: PRODUCTION HARDENING

### Week 7, Day 1-2: Caching Implementation
- [ ] Set up Redis on DigitalOcean
- [ ] Implement file hash caching in `app.py`
- [ ] Test cache hits and misses
- [ ] Verify cache invalidation works
- [ ] Monitor cache hit rate (target > 30%)

### Week 7, Day 3-4: Monitoring Setup
- [ ] Configure Sentry error tracking
- [ ] Set up DigitalOcean alerts:
  - [ ] CPU > 80% for 5 minutes
  - [ ] Memory > 90%
  - [ ] Error rate > 5%
  - [ ] Response time > 10 seconds
- [ ] Create Slack/email notifications
- [ ] Test alerts fire correctly

### Week 7, Day 5: Performance Optimization
- [ ] Enable ONNX graph optimization
- [ ] Tune tessellation parameters
- [ ] Optimize database queries
- [ ] Add connection pooling
- [ ] Implement request queuing for burst traffic

### Week 8, Day 1-3: Load Testing
- [ ] Set up k6 or locust
- [ ] Create load test scenarios:
  - [ ] 10 concurrent users
  - [ ] 50 files/hour sustained
  - [ ] Burst traffic (100 files in 10 minutes)
- [ ] Run load tests
- [ ] Analyze bottlenecks
- [ ] Optimize as needed

### Week 8, Day 4-5: Documentation & Handoff
- [ ] Create operational runbook
- [ ] Document monitoring procedures
- [ ] Document escalation procedures
- [ ] Train team on new system
- [ ] Update support documentation
- [ ] Create troubleshooting guide

**Week 7-8 Success Criteria:**
- ✅ Caching reduces 30%+ of processing load
- ✅ All alerts configured and tested
- ✅ Load tests pass (10 concurrent users)
- ✅ Error rate < 1%
- ✅ Team trained on operations

---

## POST-DEPLOYMENT (Ongoing)

### Week 9+: Production Operation
- [ ] Monitor daily metrics
- [ ] Review weekly performance reports
- [ ] Address user feedback
- [ ] Fix bugs as discovered
- [ ] Plan feature improvements

### Month 2: Optimization
- [ ] Fine-tune confidence thresholds based on real data
- [ ] Optimize for common file types
- [ ] Reduce false positives
- [ ] Improve turning feature detection

### Month 3: Model Enhancement
- [ ] Begin AAGNet training preparation
- [ ] Collect custom dataset (1000+ parts)
- [ ] Label training data
- [ ] Set up GPU environment for training

---

## ROLLBACK CHECKLIST (If Needed)

### Immediate Actions (< 5 minutes)
- [ ] Switch Edge Function URL back to old service
- [ ] Verify old service is still running
- [ ] Test with sample file
- [ ] Announce rollback to team

### Database Rollback (if needed)
- [ ] Remove new columns from `part_features`
- [ ] Restore from backup if data corrupted

### Post-Rollback
- [ ] Document what went wrong
- [ ] Fix issues in v2.0 code
- [ ] Re-test before re-deploying

---

## SUCCESS METRICS

### Technical Metrics
- [ ] 99%+ uptime
- [ ] < 5 second average response time
- [ ] < 2% error rate
- [ ] > 30% cache hit rate
- [ ] < 80% CPU utilization
- [ ] 90%+ file processing success rate

### Business Metrics
- [ ] 50-100 files processed per day
- [ ] Customer satisfaction with accuracy
- [ ] Reduction in manual review time
- [ ] Cost per file processed < $0.10

---

## FINAL VERIFICATION

### Before Marking Complete
- [ ] All code committed and pushed
- [ ] All documentation updated
- [ ] Team trained on new system
- [ ] Monitoring configured
- [ ] Backup procedures documented
- [ ] Rollback procedures tested
- [ ] Production stable for 7+ days
- [ ] No critical bugs outstanding

---

## SIGN-OFF

**Technical Lead:** _________________ Date: _________

**DevOps:** _________________ Date: _________

**Product Owner:** _________________ Date: _________

---

**Checklist Version:** 1.0.0  
**Created:** November 13, 2025  
**Last Updated:** November 13, 2025
