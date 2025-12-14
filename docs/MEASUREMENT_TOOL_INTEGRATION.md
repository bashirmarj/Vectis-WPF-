# SolidWorks-Style Measurement Tool Integration Guide

## Overview

This document provides comprehensive instructions for integrating the SolidWorks-style measurement tool into the existing Vectis CAD viewer application. The measurement tool is designed as a **completely standalone** solution that integrates seamlessly **without modifying any existing code**.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (React/Three.js)                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────────────────┐  │
│  │ SolidWorksMeasureTab│  │ SolidWorksMeasurementTool       │  │
│  │ (UI Panel)          │  │ (3D Scene Integration)          │  │
│  └─────────────────────┘  └─────────────────────────────────┘  │
│           │                           │                         │
│           └───────────┬───────────────┘                         │
│                       ▼                                         │
│         ┌─────────────────────────────┐                         │
│         │ solidworksMeasureStore      │                         │
│         │ (Zustand State Management)  │                         │
│         └─────────────────────────────┘                         │
│                       │                                         │
│                       ▼                                         │
│         ┌─────────────────────────────┐                         │
│         │ measurementApiService       │                         │
│         │ (API Communication Layer)   │                         │
│         └─────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼ REST API
┌─────────────────────────────────────────────────────────────────┐
│                  Backend (Flask/Python)                          │
├─────────────────────────────────────────────────────────────────┤
│  measurement-service/app.py                                      │
│  ├── /measure/point-to-point                                     │
│  ├── /measure/edge                                               │
│  ├── /measure/face-area                                          │
│  ├── /measure/face-to-face                                       │
│  ├── /measure/angle                                              │
│  ├── /analyze-geometry                                           │
│  └── /convert-units                                              │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
loving-roentgen/
├── measurement-service/          # Standalone backend
│   ├── app.py                    # Flask application
│   ├── requirements.txt          # Python dependencies
│   └── Dockerfile               # Container deployment
│
├── src/
│   ├── components/
│   │   └── measurement-tool/     # Frontend components
│   │       ├── index.ts          # Module exports
│   │       ├── SolidWorksMeasureTab.tsx
│   │       ├── SolidWorksMeasurementTool.tsx
│   │       └── SolidWorksMeasurementCallout.tsx
│   │
│   ├── stores/
│   │   └── solidworksMeasureStore.ts  # State management
│   │
│   └── services/
│       └── measurementApiService.ts   # API client
│
└── docs/
    └── MEASUREMENT_TOOL_INTEGRATION.md  # This file
```

## Installation

### Backend Setup

1. **Navigate to measurement service directory:**
   ```bash
   cd measurement-service
   ```

2. **Create conda environment (recommended for OpenCascade):**
   ```bash
   conda create -n measurement-service python=3.10
   conda activate measurement-service
   conda install -c conda-forge pythonocc-core=7.7.0
   pip install flask flask-cors numpy gunicorn
   ```

3. **Or use Docker:**
   ```bash
   docker build -t vectis-measurement-service .
   docker run -p 8081:8081 vectis-measurement-service
   ```

4. **Start the service:**
   ```bash
   # Development
   python app.py

   # Production
   gunicorn --bind 0.0.0.0:8081 --workers 2 --timeout 120 app:app
   ```

5. **Verify health:**
   ```bash
   curl http://localhost:8081/health
   ```

### Frontend Setup

No installation required - components are already in the source tree. Just import them where needed.

## Integration Examples

### Example 1: Basic Integration with CADViewer

Add the measurement tool to your existing CADViewer without modifying `CADViewer.tsx`:

```tsx
// MyCADViewerWrapper.tsx
import React, { useState, useRef } from 'react';
import { CADViewer } from '@/components/CADViewer';
import {
  SolidWorksMeasureTab,
  SolidWorksMeasurementTool,
  useSolidWorksMeasureStore,
} from '@/components/measurement-tool';

export const MyCADViewerWrapper: React.FC<{ meshData: MeshData }> = ({ meshData }) => {
  const [measurementEnabled, setMeasurementEnabled] = useState(false);
  const meshRef = useRef<THREE.Mesh>(null);
  const { setIsActive } = useSolidWorksMeasureStore();

  const handleMeasurementToggle = (enabled: boolean) => {
    setMeasurementEnabled(enabled);
    setIsActive(enabled);
  };

  return (
    <div className="relative w-full h-full">
      {/* Original CADViewer (unchanged) */}
      <CADViewer meshData={meshData} />

      {/* Add measurement tool button to toolbar */}
      <button
        className="absolute top-4 right-4 z-50 bg-blue-600 text-white px-4 py-2 rounded"
        onClick={() => handleMeasurementToggle(!measurementEnabled)}
      >
        {measurementEnabled ? 'Exit Measure' : 'Measure'}
      </button>

      {/* SolidWorks-style Measure Panel */}
      {measurementEnabled && (
        <SolidWorksMeasureTab
          onClose={() => handleMeasurementToggle(false)}
          isMeasuring={measurementEnabled}
        />
      )}
    </div>
  );
};
```

### Example 2: Add to Canvas Scene

```tsx
// Inside your Canvas component
import { SolidWorksMeasurementTool } from '@/components/measurement-tool';

<Canvas>
  {/* Existing scene content */}
  <MeshModel ref={meshRef} meshData={meshData} />

  {/* Add measurement tool */}
  <SolidWorksMeasurementTool
    meshData={meshData}
    meshRef={meshRef.current?.mesh || null}
    featureEdgesGeometry={meshRef.current?.featureEdgesGeometry || null}
    enabled={measurementEnabled}
    boundingSphere={{
      center: boundingBox.center,
      radius: Math.max(boundingBox.width, boundingBox.height, boundingBox.depth) / 2,
    }}
  />
</Canvas>
```

### Example 3: Use Store Directly

```tsx
import { useSolidWorksMeasureStore, createMeasurement } from '@/components/measurement-tool';

const MyComponent = () => {
  const {
    measurements,
    unitSystem,
    setUnitSystem,
    addMeasurement,
    clearAllMeasurements,
  } = useSolidWorksMeasureStore();

  // Toggle units
  const toggleUnits = () => {
    setUnitSystem(unitSystem === 'metric' ? 'imperial' : 'metric');
  };

  // Add custom measurement
  const addCustomMeasurement = () => {
    const measurement = createMeasurement('point_to_point', 42.5, {
      deltaX: 30.0,
      deltaY: 25.0,
      deltaZ: 10.5,
      point1: new THREE.Vector3(0, 0, 0),
      point2: new THREE.Vector3(30, 25, 10.5),
    });
    addMeasurement(measurement);
  };

  return (
    <div>
      <p>Total measurements: {measurements.length}</p>
      <button onClick={toggleUnits}>
        Units: {unitSystem === 'metric' ? 'mm' : 'in'}
      </button>
      <button onClick={addCustomMeasurement}>Add Measurement</button>
      <button onClick={clearAllMeasurements}>Clear All</button>
    </div>
  );
};
```

### Example 4: Use API Service

```tsx
import { measurementApi, localCalculations } from '@/services/measurementApiService';

// Measure point-to-point via backend
const result = await measurementApi.measurePointToPoint(
  [0, 0, 0],
  [10, 20, 30],
  'metric'
);

console.log(result.value);        // 37.416...
console.log(result.display_value); // "37.416 mm"
console.log(result.delta_x);       // 10
console.log(result.delta_y);       // 20
console.log(result.delta_z);       // 30

// Or use local calculations (no backend required)
const localResult = localCalculations.pointToPoint(
  new THREE.Vector3(0, 0, 0),
  new THREE.Vector3(10, 20, 30),
  'metric'
);
```

## API Reference

### Backend Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "vectis-measurement-service",
  "version": "1.0.0",
  "capabilities": ["point_to_point", "edge_measurement", ...],
  "timestamp": "2025-12-14T10:30:00Z"
}
```

#### `POST /measure/point-to-point`
Calculate distance between two points with XYZ deltas.

**Request:**
```json
{
  "point1": [0, 0, 0],
  "point2": [10, 20, 30],
  "unit_system": "metric"
}
```

**Response:**
```json
{
  "success": true,
  "measurement_type": "point_to_point",
  "value": 37.416573867739416,
  "unit": "mm",
  "label": "Distance: 37.417 mm",
  "display_value": "37.417 mm",
  "delta_x": 10.0,
  "delta_y": 20.0,
  "delta_z": 30.0,
  "point1": [0, 0, 0],
  "point2": [10, 20, 30],
  "confidence": 1.0,
  "backend_match": true,
  "processing_time_ms": 1
}
```

#### `POST /measure/edge`
Measure an edge (length, diameter, or radius).

**Request:**
```json
{
  "edge_info": {
    "edge_id": 0,
    "edge_type": "circle",
    "start_point": [10, 0, 0],
    "end_point": [10, 0, 0],
    "diameter": 20.0
  },
  "unit_system": "metric"
}
```

#### `POST /measure/face-to-face`
Calculate distance and angle between two faces.

**Request:**
```json
{
  "face1": {
    "center": [0, 0, 0],
    "normal": [0, 0, 1]
  },
  "face2": {
    "center": [0, 0, 10],
    "normal": [0, 0, 1]
  },
  "unit_system": "metric"
}
```

#### `POST /analyze-geometry`
Analyze STEP file for measurements.

**Request:** `multipart/form-data` with `file` field

**Response:**
```json
{
  "success": true,
  "file_hash": "abc123...",
  "face_count": 24,
  "edge_count": 48,
  "faces": [...],
  "edges": [...],
  "processing_time_ms": 150
}
```

#### `POST /convert-units`
Convert between unit systems.

**Request:**
```json
{
  "value": 25.4,
  "from_unit": "mm",
  "to_unit": "in",
  "decimals": 3
}
```

**Response:**
```json
{
  "success": true,
  "original_value": 25.4,
  "original_unit": "mm",
  "converted_value": 1.0,
  "converted_unit": "in",
  "display_value": "1.000 in"
}
```

### Frontend Store API

```typescript
interface SolidWorksMeasureStore {
  // State
  measurements: SWMeasurement[];
  activeMeasurement: SWMeasurement | null;
  unitSystem: 'metric' | 'imperial';
  showDeltas: boolean;
  precision: number;

  // Actions
  addMeasurement: (measurement: SWMeasurement) => void;
  removeMeasurement: (id: string) => void;
  clearAllMeasurements: () => void;
  setUnitSystem: (system: UnitSystem) => void;
  setShowDeltas: (show: boolean) => void;
  setPrecision: (precision: number) => void;

  // Undo/Redo
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;
}
```

## Measurement Types

| Type | Description | Display |
|------|-------------|---------|
| `point_to_point` | Distance between two points | `42.500 mm` with ΔX, ΔY, ΔZ |
| `edge_length` | Length of a line edge | `25.000 mm` |
| `edge_diameter` | Diameter of a circular edge | `Ø 12.000 mm` |
| `edge_radius` | Radius of an arc edge | `R 6.000 mm` |
| `face_area` | Surface area of a face | `150.000 mm²` |
| `face_to_face` | Distance between faces | `10.000 mm (0.0°)` |
| `angle` | Angle between vectors | `45.00°` |

## UI/UX Guidelines

### SolidWorks-Style Colors
- **Primary Blue:** `#1a5fb4`
- **Edge Hover:** `#ff8800` (orange)
- **Distance Lines:** `#1a5fb4` (blue)
- **Face Measurements:** `#00AA55` (green)

### Interaction Patterns
1. **Hover over edge:** Shows orange highlight + measurement preview
2. **Click on edge:** Creates instant diameter/radius/length measurement
3. **Click on face point 1:** Places first marker
4. **Click on face point 2:** Creates distance + angle measurement
5. **Press M key:** Toggle measurement mode
6. **Press Escape:** Cancel current measurement

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `M` | Toggle measurement mode |
| `Escape` | Cancel/exit measurement |
| `Ctrl+Z` | Undo last measurement |
| `Ctrl+Y` | Redo measurement |

## Performance Considerations

1. **Use local calculations when possible** - The `localCalculations` module provides fast client-side calculations without backend calls.

2. **Backend for complex geometry** - Use the backend API for face-to-face distances and area calculations that require exact CAD geometry.

3. **Geometry caching** - The `/analyze-geometry` endpoint caches results by file hash for repeated measurements.

4. **Lazy loading** - Import measurement components only when needed:
   ```tsx
   const SolidWorksMeasureTab = React.lazy(() =>
     import('@/components/measurement-tool').then(m => ({ default: m.SolidWorksMeasureTab }))
   );
   ```

## Troubleshooting

### Backend not responding
1. Check if the service is running: `curl http://localhost:8081/health`
2. Verify port 8081 is not blocked
3. Check logs for OpenCascade errors

### Measurements not appearing
1. Ensure `enabled={true}` prop is set
2. Check that `meshRef` is properly forwarded
3. Verify `tagged_edges` exists in meshData

### Unit conversion issues
1. Default is metric (mm)
2. Use `setUnitSystem('imperial')` for inches
3. Precision can be set 0-6 decimals

## Extending the Tool

### Add New Measurement Type

1. **Update store:**
   ```typescript
   // solidworksMeasureStore.ts
   type MeasurementType = ... | 'volume' | 'mass';
   ```

2. **Add API endpoint:**
   ```python
   # app.py
   @app.route('/measure/volume', methods=['POST'])
   def measure_volume():
       # Implementation
   ```

3. **Add API client method:**
   ```typescript
   // measurementApiService.ts
   async measureVolume(faceIds: number[]): Promise<MeasurementResultApi>
   ```

4. **Update visualization:**
   ```tsx
   // SolidWorksMeasurementTool.tsx
   if (measurement.type === 'volume') {
     // Render volume measurement
   }
   ```

## License

This measurement tool is part of the Vectis Machining Platform and is subject to the same license terms.
