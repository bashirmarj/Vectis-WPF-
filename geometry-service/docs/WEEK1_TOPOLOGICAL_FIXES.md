# Week 1: Topological Fixes Implementation Plan

**Goal**: Implement universal topological recognition to achieve parity with Analysis Situs

**Timeline**: 7 days  
**Target**: Fix holes, fillets, pockets using topology-based methods (not heuristics)

---

## 🎯 Executive Summary

### Current State (57.1% parity)
✅ **Working**:
- Structure validation: PASS
- Chamfers: 0 = 0 (PASS)
- Steps: 1 = 1 (PASS) - Perfect match!
- Threads: 0 = 0 (PASS)

❌ **Broken** (requires topological fixes):
- **Holes**: Expected 22, Got 0 (100% missing)
- **Pockets**: Expected 5, Got 2 (60% missing)
- **Fillets**: Expected 11, Got 0 (100% missing)

### Root Causes Identified

| Feature | Root Cause | Current Approach | AS Approach |
|---------|-----------|------------------|-------------|
| **Holes** | Missing coaxial grouping | Filters individual cylinders | Groups coaxial cylinders first |
| **Fillets** | Graph format mismatch | Expects `GraphNode` objects | Uses consistent dict format |
| **Pockets** | Incomplete boundary detection | Uses area thresholds/heuristics | Traces closed boundary loops |

---

## 🛣️ Week 1 Roadmap

### Day 1-2: Foundation Layer
**Objective**: Add topological primitives to `AAGGraphBuilder`

```python
# NEW METHODS TO ADD TO graph_builder.py

class AAGGraphBuilder:
    
    def trace_boundary_loop(self, start_edge, face) -> List[Edge]:
        """Trace closed boundary loop around a face."""
        pass
    
    def group_coaxial_cylinders(self, cylinders: List[Node]) -> List[List[Node]]:
        """Group cylinders by axis alignment."""
        pass
    
    def get_wall_faces(self, bottom_face: Node) -> List[Node]:
        """Get vertical wall faces adjacent to bottom face."""
        pass
    
    def is_closed_boundary(self, edges: List[Edge]) -> bool:
        """Check if edges form closed loop."""
        pass
    
    def check_wall_orientation(self, walls: List[Node], 
                               bottom: Node) -> bool:
        """Check if all walls point away from bottom."""
        pass
```

**Deliverables**:
- [ ] `trace_boundary_loop()` method
- [ ] `group_coaxial_cylinders()` method  
- [ ] Unit tests for new methods
- [ ] Documentation with examples

**Acceptance Criteria**:
- `trace_boundary_loop()` correctly handles open vs closed loops
- `group_coaxial_cylinders()` groups by axis within 1° tolerance
- Tests pass on synthetic geometry

---

### Day 3-4: Hole Recognizer Refactor
**Objective**: Implement group-then-classify architecture

#### Current Architecture (❌)
```python
def recognize_holes(graph):
    holes = []
    for cylinder in graph.get_cylinders():
        if is_hole_sized(cylinder):  # ❌ Size filter
            if has_end_faces(cylinder):  # ❌ Heuristic
                holes.append(cylinder)
    return holes
```

#### New Architecture (✅)
```python
def recognize_holes(graph):
    # 1. Group coaxial cylinders
    cylinder_groups = graph.group_coaxial_cylinders(
        graph.get_cylinders()
    )
    
    holes = []
    for group in cylinder_groups:
        # 2. Classify entire group
        if is_hole_pattern(group, graph):
            hole = build_compound_hole(group, graph)
            holes.append(hole)
        # else: might be shaft, protrusion, etc.
    
    return holes

def is_hole_pattern(group: List[Node], graph) -> bool:
    """Topological test for hole pattern."""
    # Check if cylinders connect two parallel faces
    start_faces = [f for f in graph.neighbors(group[0]) 
                   if f.shape_type == 'plane']
    end_faces = [f for f in graph.neighbors(group[-1])
                 if f.shape_type == 'plane']
    
    # Holes have entry and exit faces
    if not start_faces or not end_faces:
        return False
    
    # Check parallelism
    return are_parallel(start_faces[0].normal, 
                       end_faces[0].normal)

def build_compound_hole(group: List[Node], graph) -> dict:
    """Build hole from coaxial cylinder group."""
    # Largest cylinder = main hole
    main = max(group, key=lambda c: c.radius)
    
    # Smaller cylinders = counterbores/countersinks
    features = []
    for cyl in sorted(group, key=lambda c: -c.radius):
        if cyl != main:
            features.append({
                'type': classify_bore_type(cyl, main),
                'diameter': cyl.radius * 2,
                'depth': cyl.height
            })
    
    return {
        'type': 'hole',
        'diameter': main.radius * 2,
        'depth': main.height,
        'features': features,  # counterbores, countersinks
        'face_ids': [c.face_id for c in group]
    }
```

**Deliverables**:
- [ ] Refactored `hole_recognizer.py` with new architecture
- [ ] Remove ALL size-based filters
- [ ] Implement `is_hole_pattern()` topological test
- [ ] Implement `build_compound_hole()` for multi-diameter holes
- [ ] Unit tests on parts with counterbores

**Acceptance Criteria**:
- Recognizes all 22 holes in FreeCAD Beginner 163 part
- Handles through-holes, blind holes, counterbores, countersinks
- Works on ANY part (not tuned to one part)

---

### Day 5: Fillet Recognizer Fix
**Objective**: Standardize graph format

#### Problem
Recognizer crashes because:
```python
# graph_builder.py creates dicts:
graph.add_node({'face_id': 42, 'shape_type': 'toroid'})

# fillet_recognizer.py expects objects:
if node.shape_type == 'toroid':  # ❌ AttributeError
```

#### Solution
Pick ONE format and standardize:

**Option A: Use Typed Objects**
```python
# graph_builder.py
from dataclasses import dataclass

@dataclass
class GraphNode:
    face_id: int
    shape_type: str
    area: float
    # ... other attributes

# Convert at entry point
def build_graph(brep) -> Graph:
    graph = Graph()
    for face in brep.faces():
        node = GraphNode(
            face_id=face.id,
            shape_type=classify_shape(face),
            area=compute_area(face)
        )
        graph.add_node(node)
    return graph
```

**Option B: Use Dicts Everywhere**
```python
# Update ALL recognizers to use dict notation:
if node['shape_type'] == 'toroid':
    radius = node['radius']
```

**Recommendation**: Use **typed objects** for better IDE support and type safety.

**Deliverables**:
- [ ] Choose ONE graph format
- [ ] Update `AAGGraphBuilder` to use chosen format
- [ ] Update ALL recognizers to match
- [ ] Add type hints throughout
- [ ] Verify fillet recognizer runs without errors

**Acceptance Criteria**:
- Fillet recognizer processes all candidate faces
- Recognizes all 11 fillets in test part
- No type/format errors

---

### Day 6: Pocket Recognizer Refactor
**Objective**: Implement topological boundary detection

#### Current Architecture (❌)
```python
def recognize_pockets(graph):
    pockets = []
    for face in graph.get_planes():
        if is_depressed(face):  # Heuristic
            if face.area > MIN_AREA:  # ❌ Magic number
                if aspect_ratio(face) < MAX_RATIO:  # ❌ Magic number
                    pockets.append(face)
    return pockets
```

#### New Architecture (✅)
```python
def recognize_pockets(graph):
    pockets = []
    
    for candidate in graph.get_planes():
        # 1. Trace boundary
        boundary = graph.trace_boundary_loop(
            start_edge=candidate.edges[0],
            face=candidate
        )
        
        # 2. Check if closed
        if not graph.is_closed_boundary(boundary):
            continue
        
        # 3. Get wall faces
        walls = graph.get_wall_faces(candidate)
        
        # 4. Check wall orientation
        if not graph.check_wall_orientation(walls, candidate):
            continue
        
        # 5. Build pocket
        pocket = {
            'type': 'pocket',
            'bottom_face_id': candidate.face_id,
            'wall_face_ids': [w.face_id for w in walls],
            'boundary_length': sum(e.length for e in boundary),
            'area': candidate.area,
            'depth': compute_depth(candidate, walls)
        }
        pockets.append(pocket)
    
    return pockets

def compute_depth(bottom: Node, walls: List[Node]) -> float:
    """Compute pocket depth from geometry."""
    # Find top faces of walls
    top_faces = []
    for wall in walls:
        adjacent = [n for n in graph.neighbors(wall)
                   if n.shape_type == 'plane' and n != bottom]
        top_faces.extend(adjacent)
    
    if not top_faces:
        return 0.0
    
    # Depth = distance from bottom to top
    top_z = max(f.centroid.z for f in top_faces)
    bottom_z = bottom.centroid.z
    return abs(top_z - bottom_z)
```

**Deliverables**:
- [ ] Refactored `pocket_recognizer.py` with boundary tracing
- [ ] Remove ALL area/aspect ratio filters
- [ ] Implement topological depth computation
- [ ] Handle irregular pocket shapes

**Acceptance Criteria**:
- Recognizes all 5 pockets in test part
- Handles small, large, irregular pockets equally
- Works on multi-level pockets

---

### Day 7: Integration & Multi-Part Validation
**Objective**: Verify fixes work universally

**Test Suite**:
1. FreeCAD Beginner 163 (prismatic milling)
2. Shaft with Keyways (lathe + milling)
3. Complex bracket (multi-level pockets)
4. Cylindrical part with counterbored holes
5. Part with variable-radius fillets

**Validation Process**:
```bash
# Run validation on all test parts
for part in test_parts/*.step; do
    python tests/validation/enhanced_validator_with_logging.py \
        "$part" \
        "ground_truth/${part%.step}.json"
done

# Check pass rates
python tests/validation/analyze_multi_part_results.py
```

**Deliverables**:
- [ ] Run validation on 5+ diverse parts
- [ ] Document pass rates per part
- [ ] Fix any universal issues found
- [ ] Generate final validation report

**Acceptance Criteria**:
- **>90% parity** on ALL test parts (not just one)
- No part-specific tweaks (universal recognition)
- Clean validation reports with detailed logs

---

## 📊 Success Metrics

### Week 1 End Goals

| Metric | Current | Target |
|--------|---------|--------|
| **Overall Parity** | 57.1% | **>90%** |
| **Holes** | 0/22 (0%) | **22/22 (100%)** |
| **Pockets** | 2/5 (40%) | **5/5 (100%)** |
| **Fillets** | 0/11 (0%) | **11/11 (100%)** |
| **Test Parts** | 2 | **5+** |
| **Universal Recognition** | No | **Yes** |

### Code Quality
- [ ] All new code has type hints
- [ ] All new methods have docstrings
- [ ] All new code has unit tests
- [ ] No magic numbers (thresholds, filters)
- [ ] Uses topological queries, not heuristics

---

## 🛠️ Implementation Guidelines

### Architectural Principles

#### 1. **Use Topological Queries, Not Heuristics**

❌ **Wrong (Part-Specific)**:
```python
if area > 500: return False
if aspect_ratio < 0.5: return False
if radius < 1.0: return False
```

✅ **Right (Universal)**:
```python
if is_closed_boundary(edges): return True
if connects_parallel_faces(cylinder): return True
if has_wall_orientation(faces): return True
```

#### 2. **Group Before Classify**

❌ **Wrong**:
```python
for cylinder in cylinders:
    if looks_like_hole(cylinder):  # Individual decision
        add_hole()
```

✅ **Right**:
```python
groups = group_coaxial(cylinders)  # Group first
for group in groups:
    if is_hole_pattern(group):  # Classify group
        add_hole()
```

#### 3. **Consistent Data Structures**

✅ **Right**:
```python
# ALL recognizers use same graph format
# Convert ONCE at entry point
class AAGGraphBuilder:
    def build_graph(self, brep) -> Graph[GraphNode]:
        # Convert to consistent format here
        pass
```

### Testing Strategy

**Unit Tests** (per method):
```python
def test_trace_boundary_loop_closed():
    """Test boundary tracing on closed loop."""
    graph = create_square_pocket_graph()
    boundary = graph.trace_boundary_loop(start_edge, bottom_face)
    assert is_closed_boundary(boundary)
    assert len(boundary) == 4  # Square has 4 edges

def test_group_coaxial_cylinders():
    """Test cylinder grouping by axis."""
    graph = create_counterbored_hole_graph()
    groups = graph.group_coaxial_cylinders(graph.get_cylinders())
    assert len(groups) == 1  # All coaxial
    assert len(groups[0]) == 2  # Main hole + counterbore
```

**Integration Tests** (per recognizer):
```python
def test_hole_recognizer_on_test_part():
    """Test hole recognizer on FreeCAD Beginner 163."""
    graph = load_graph('tests/fixtures/FreeCAD_Beginner_163-Body.step')
    recognizer = HoleRecognizer()
    holes = recognizer.recognize(graph)
    assert len(holes) == 22  # Expected from AS
```

**System Tests** (full validation):
```python
def test_complete_validation_multi_part():
    """Test complete validation on multiple parts."""
    for part_name in ['part1', 'part2', 'part3']:
        report = run_validation(part_name)
        assert report.pass_rate > 90.0
```

---

## 📝 Daily Checklist Template

### Daily Standup Questions
1. What did I accomplish yesterday?
2. What am I working on today?
3. Any blockers?

### Daily End-of-Day
- [ ] Code committed and pushed
- [ ] Tests passing
- [ ] Documentation updated
- [ ] No regressions on existing parts
- [ ] Tomorrow's work planned

### Example Daily Log

**Day 1 - Monday**
- ✅ Implemented `trace_boundary_loop()` in graph_builder.py
- ✅ Added unit tests for closed/open loop cases
- ✅ Tested on synthetic square pocket - works!
- 🚧 Need to handle edge cases (holes in boundary)

**Day 2 - Tuesday**
- ✅ Implemented `group_coaxial_cylinders()`
- ✅ Added tolerance parameter (1° default)
- ✅ Tested on counterbored hole - groups correctly!
- ✅ Documented algorithm in docstring

---

## 🚦 Risk Management

### Potential Blockers

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Graph format changes break existing code** | High | High | Write adapter layer, gradual migration |
| **Boundary tracing fails on complex geometry** | Medium | High | Start with simple cases, add complexity |
| **Coaxial grouping too strict/loose** | Medium | Medium | Make tolerance configurable, test range |
| **Validation takes too long** | Low | Low | Run on subset first, parallelize if needed |

### Contingency Plans

**If behind schedule by Day 3**:
- Focus on ONE feature (holes) and perfect it
- Defer fillet/pocket fixes to Week 2
- Ensure at least holes reach 100% parity

**If validation fails on new parts**:
- Analyze failure modes
- Add edge case handling
- DO NOT add part-specific fixes
- Find universal topological solution

---

## 🚀 Getting Started

### Setup

1. **Create feature branch**:
```bash
git checkout -b week1-topological-fixes
```

2. **Run baseline validation**:
```bash
python tests/validation/enhanced_validator_with_logging.py \
    tests/fixtures/FreeCAD_Beginner_163-Body.step \
    tests/fixtures/analysis_situs_log.json
```

3. **Review current recognizers**:
```bash
# Understand current implementations
cat geometry-service/aag_pattern_engine/recognizers/hole_recognizer.py
cat geometry-service/aag_pattern_engine/recognizers/fillet_chamfer_recognizer.py
cat geometry-service/aag_pattern_engine/recognizers/pocket_recognizer.py
```

4. **Start with Day 1 tasks** (see above)

### Workflow

```bash
# 1. Implement feature
vim geometry-service/aag_pattern_engine/graph_builder.py

# 2. Add tests
vim geometry-service/tests/unit/test_graph_builder.py

# 3. Run tests
pytest tests/unit/test_graph_builder.py -v

# 4. Run validation
python tests/validation/enhanced_validator_with_logging.py ...

# 5. Commit
git add -A
git commit -m "feat: add trace_boundary_loop() to graph builder"
git push origin week1-topological-fixes
```

---

## 📚 References

### Key Documents
- [Analysis Situs Output Structure](../tests/fixtures/README.md)
- [Graph Builder Documentation](../aag_pattern_engine/graph_builder.py)
- [Validation Suite Guide](../tests/validation/README.md)

### Analysis Situs Comparison
- **Their approach**: Group → Classify → Validate
- **Our new approach**: Group → Classify → Validate (match exactly)
- **Key insight**: They use topology, not heuristics

### Useful Papers
- "Feature Recognition in CAD Models" (Vandenbrande & Requicha, 1993)
- "Graph-based Feature Recognition" (Kim & Wang, 2002)
- Analysis Situs documentation: https://analysissitus.org

---

## ❓ FAQ

### Q: Should I remove ALL filters/thresholds?
A: **Yes**. If Analysis Situs doesn't use them, neither should we. Use topological tests instead.

### Q: What if my fix works on one part but breaks another?
A: **That's a red flag**. Find the universal topological principle that works on BOTH parts.

### Q: How strict should coaxial grouping be?
A: Start with **1° tolerance**. Make it configurable. Test on parts with slightly misaligned holes.

### Q: Should I optimize for speed?
A: **No**. Focus on correctness first. Optimization is Week 3+.

### Q: What if I find a bug in Analysis Situs output?
A: **Document it** but match their output anyway. We can file upstream bug later.

---

## ✅ Week 1 Completion Checklist

Before moving to Week 2, ensure:

- [ ] All three recognizers refactored (holes, fillets, pockets)
- [ ] All magic numbers removed
- [ ] Graph format standardized
- [ ] Boundary tracing implemented
- [ ] Coaxial grouping implemented
- [ ] ≥ 90% parity on FreeCAD Beginner 163 part
- [ ] ≥ 85% parity on 4+ other test parts
- [ ] All tests passing
- [ ] Code reviewed and documented
- [ ] Validation reports generated and committed
- [ ] No regressions on existing parts

**Sign-off**: Ready for Week 2 (Multi-Part Validation) ✓

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-20  
**Owner**: VectisMachining Team  
**Status**: 🟢 Active
