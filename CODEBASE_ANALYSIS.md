# Vectis-WPF Codebase Analysis

## Project Overview

**Vectis Manufacturing** is a full-stack web-based CAD analysis, quotation, and manufacturing management platform. It enables customers to upload 3D CAD files (STEP/IGES), perform real-time 3D visualization and measurement analysis, receive instant DFM (Design for Manufacturing) analysis and preliminary quotes, chat with AI engineering assistants, and manage projects and quotation requests.

The platform handles custom manufacturing services including CNC machining, sheet metal, die casting, wire EDM, heat treatment, and assembly.

---

## Architecture

The system follows a **microservices architecture**:

```
Frontend (React/Vite/TypeScript)
    |
    v
Supabase Edge Functions (API Layer / Serverless)
    |
    v
Backend Services (Python/Flask):
    - Geometry Analysis Service (CAD parsing, feature recognition)
    - Measurement Service (dimensional analysis)
    - Mesh Service (tessellation)
    |
    v
Supabase PostgreSQL (Database / Auth / Storage)
```

### Key Patterns
- **MVVM-like**: React components as Views, Zustand stores as ViewModel state, custom hooks for business logic
- **Context API**: `AuthContext` for user/session management
- **Command Pattern**: Measurement undo/redo system
- **Service Layer**: `measurementApiService.ts` for API communication
- **Compound Components**: CADViewer composed of MeshModel, MeasurementPanel, DimensionAnnotations, etc.

---

## Technology Stack

### Frontend
| Category | Technology |
|----------|-----------|
| Framework | React 18.3.1 |
| Build Tool | Vite 5.4.19 |
| Language | TypeScript 5.8.3 |
| UI Components | shadcn-ui (49 Radix UI components) |
| Styling | Tailwind CSS 3.4.17 |
| 3D Graphics | Three.js 0.170.0, react-three/fiber 8.18.0, react-three/drei 9.122.0 |
| State Management | Zustand 5.0.8, TanStack React Query 5.83.0 |
| Routing | React Router 6.30.1 |
| Animation | Framer Motion 12.23.26, React Spring 9.7.5 |
| Forms | React Hook Form 7.61.1, Zod 3.25.76 |
| Database Client | @supabase/supabase-js 2.58.0 |

### Backend
| Category | Technology |
|----------|-----------|
| Runtime | Python 3.10 |
| Web Framework | Flask 3.1.0 |
| CAD Processing | pythonocc-core 7.7.2 (OpenCascade) |
| Data Processing | NumPy 1.24.4, NetworkX 3.2.1 |
| Server | Gunicorn 21.2.0 |
| Cloud | Supabase 2.7.4 |

### Infrastructure
| Category | Technology |
|----------|-----------|
| Database | Supabase PostgreSQL |
| Auth | Supabase Auth |
| Storage | Supabase Storage |
| Edge Functions | Supabase Functions (Deno/TypeScript) |
| Containerization | Docker (Miniconda 3 base) |
| CI/CD | GitHub Actions |
| Registry | GitHub Container Registry (ghcr.io) |

---

## Project Structure

```
Vectis-WPF-/
├── src/                           # React frontend source
│   ├── main.tsx                   # Entry point
│   ├── App.tsx                    # Root app (routing, providers)
│   ├── components/
│   │   ├── cad-viewer/            # 3D CAD visualization (25 files)
│   │   ├── chat/                  # AI engineering chat
│   │   ├── dashboard/             # Customer dashboard
│   │   ├── home/                  # Homepage sections
│   │   ├── measurement-tool/      # SolidWorks measurement integration
│   │   ├── part-upload/           # File upload flow
│   │   ├── ui/                    # shadcn-ui components (49 files)
│   │   ├── admin/                 # Admin panel
│   │   └── validation/            # CAD validation reports
│   ├── pages/                     # Route pages (11 routes)
│   ├── hooks/                     # Custom React hooks
│   ├── contexts/                  # React Context (Auth)
│   ├── stores/                    # Zustand stores (measurement state)
│   ├── services/                  # API services
│   ├── lib/                       # Utility libraries
│   └── integrations/supabase/     # Supabase client + types
│
├── geometry-service/              # Python CAD analysis microservice
│   ├── app.py                     # Flask app (port 5000)
│   ├── aag_pattern_engine/        # AAG-based feature recognition
│   ├── occwl/                     # OpenCascade wrapper library
│   ├── tests/                     # Feature recognition tests
│   ├── Dockerfile                 # Production container
│   └── requirements.txt
│
├── measurement-service/           # Python measurement microservice
│   ├── app.py                     # Flask app (port 8081)
│   ├── Dockerfile
│   └── requirements.txt
│
├── mesh-service/                  # Mesh tessellation microservice
│   ├── mesh_service.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── supabase/                      # Backend infrastructure
│   ├── config.toml                # Supabase configuration
│   ├── migrations/                # Database migrations
│   └── functions/                 # 12 Edge Functions
│       ├── analyze-cad/
│       ├── analyze-measurement/
│       ├── calculate-preliminary-quote/
│       ├── engineering-chat/
│       ├── explain-features/
│       └── ... (7 more)
│
├── Legacy/                        # Deprecated services
├── docs/                          # Documentation
├── .github/workflows/             # CI/CD pipelines
└── [config files]                 # vite, tailwind, tsconfig, eslint, etc.
```

---

## Key Entry Points

### Frontend
- **Application Entry**: `src/main.tsx` - Renders React root
- **App Root**: `src/App.tsx` - Router + Providers (QueryClient, Tooltip, Auth)

### Routes
| Path | Page | Description |
|------|------|-------------|
| `/` | Index | Homepage |
| `/about` | About | Company information |
| `/services` | Services | Manufacturing capabilities |
| `/contact` | Contact | Contact form |
| `/auth` | Auth | Login/registration |
| `/admin` | Admin | Admin dashboard |
| `/admin/quotations/:id` | QuotationDetails | Quote management |
| `/admin/pricing-settings` | PricingSettings | Price configuration |
| `/admin/setup` | AdminSetup | Initial admin setup |
| `/validation` | Validation | CAD file validation |
| `/dashboard` | CustomerDashboard | Customer project view |

### Backend Services
| Service | Port | Main File | Endpoints |
|---------|------|-----------|-----------|
| Geometry | 5000 | `geometry-service/app.py` | `/health`, `/analyze-cad` |
| Measurement | 8081 | `measurement-service/app.py` | Measurement endpoints |
| Mesh | - | `mesh-service/mesh_service.py` | Mesh tessellation |

---

## Key Components

### CAD Viewer (`src/components/cad-viewer/`)
The 3D visualization system consists of 25+ files:
- `CADViewer.tsx` - Main Three.js canvas and scene
- `MeshModel.tsx` - 3D mesh rendering
- `DimensionAnnotations.tsx` - Measurement overlays
- `MeasurementPanel.tsx` / `AdvancedMeasurementTool.tsx` - Measurement UI
- `UnifiedCADToolbar.tsx` - Toolbar controls
- `OrientationCube_UNIFIED.tsx` - View orientation cube
- `SilhouetteEdges.tsx` - Edge rendering
- `enhancements/` - Lighting, materials, post-processing effects

### Feature Recognition (`geometry-service/aag_pattern_engine/`)
AAG (Attributed Adjacency Graph) based feature recognition system:
- `graph_builder.py` - Builds adjacency graphs from B-Rep
- `pattern_matcher.py` - Identifies manufacturing features
- `geometric_recognizer.py` - Geometric feature classification
- `feature_validator.py` - Validation against ground truth
- `tool_accessibility_analyzer.py` - Machining accessibility analysis
- `machining_configuration_detector.py` - Setup/configuration detection

### Engineering Chat (`src/components/chat/EngineeringChatWidget.tsx`)
AI-powered chat interface for manufacturing questions, powered by Supabase Edge Functions calling AI APIs with manufacturing domain knowledge.

### Part Upload Flow (`src/components/part-upload/`)
Multi-step upload wizard:
1. `FileUploadScreen.tsx` - CAD file upload
2. `ModeSelector.tsx` - Analysis mode selection
3. `PartConfigScreen.tsx` - Part configuration
4. `ProjectSelector.tsx` - Project assignment
5. `RoutingEditor.tsx` - Manufacturing routing

---

## State Management

### Zustand Stores
- `measurementStore.ts` (256 lines) - Measurement UI state with undo/redo
- `solidworksMeasureStore.ts` - SolidWorks-style measurement state

### React Context
- `AuthContext.tsx` (135 lines) - User authentication, session, role management

### React Query
- Data fetching for customer projects, parts, measurements
- Server state caching and synchronization

---

## Testing

### Backend (Python)
Located in `geometry-service/tests/`:
- `test_complete_analysis_situs_parity.py` - Feature recognition validation against Analysis Situs ground truth
- `verify_all_fixes.py` - Comprehensive fix verification
- `verify_fillet_fixes.py` - Fillet recognition tests
- `verify_hole_fixes.py` - Hole recognition tests
- `verify_volume_pipeline.py` - Volume decomposition verification
- `validation/` - Validation utilities (models, parsers, validators)

### Frontend
No frontend unit tests configured. ESLint is set up for static analysis.

---

## Deployment

### Docker Configuration
- **Geometry Service**: Miniconda3 base, Python 3.10, pythonocc-core 7.7.2, 2 Gunicorn workers, port 5000
- **Measurement Service**: Miniconda3 base, pythonocc-core 7.7.0, port 8081
- **Mesh Service**: Lightweight Flask container

### CI/CD
- GitHub Actions workflow (`.github/workflows/build-base-image.yml`) builds Docker base images on dependency changes
- Images pushed to GitHub Container Registry (ghcr.io)

### Deployment Options
1. Railway (~$5-10/month) - Recommended
2. Render (Free tier available)
3. DigitalOcean App Platform ($5/month)
4. Docker self-hosted

---

## Database Schema (Supabase)
- `user_roles` - Admin role tracking
- `quotation_submissions` - Quote requests
- `customer_projects` - Project management
- `parts` - CAD file records
- `measurements` - Measurement history

---

## Notable Design Decisions

1. **Feature Recognition Toggle**: `SKIP_FEATURE_RECOGNITION = True` in `geometry-service/app.py` - Full recognition takes 60+ seconds; when skipped, only tessellation + basic metrics (5-15s)
2. **Microservices Split**: Geometry, measurement, and mesh services are independently deployable and scalable
3. **WASM Support**: Vite configured with WASM plugin for `occt-import-js` (client-side STEP import)
4. **Measurement Undo/Redo**: Command pattern implementation in Zustand store
5. **Serverless API Layer**: Supabase Edge Functions handle coordination, AI chat, quotes, and email notifications

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| TypeScript Files | ~148 |
| TypeScript Lines | ~10,549 |
| UI Components (shadcn) | 49 |
| Custom React Components | 40+ |
| Python Services | 3 |
| Supabase Edge Functions | 12 |
| npm Dependencies | 102+ |
| Frontend Routes | 11 |
| Custom Animations | 20+ |
