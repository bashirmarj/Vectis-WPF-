# Vectis Machining - Development Guidelines

## Constitution

Read `.specify/memory/constitution.md` before making any architectural or feature decisions.
The constitution defines non-negotiable principles, scope rules, and quality gates.

**Key constraints from the constitution:**

- RFQ submission is the core product — optimize for it above everything else
- Near-term focus: CAD viewer + feature recognition (80%+ accuracy) + RFQ submit flow
- UX is the top quality priority
- Users MUST sign in before uploading files
- No secrets in client code
- Defer anything that doesn't unblock the near-term focus

## Tech Stack

- **Frontend**: React 18 + Vite + TypeScript + Tailwind CSS + shadcn/ui
- **3D Viewer**: Three.js + react-three/fiber + react-three/drei + occt-import-js (WASM)
- **State**: Zustand + TanStack React Query
- **Routing**: React Router 6
- **Auth/DB/Storage**: Supabase (Auth, PostgreSQL, Storage, Edge Functions)
- **Backend Services**: Python 3.10 + Flask (geometry-service, measurement-service, mesh-service)
- **CAD Processing**: pythonocc-core (OpenCascade) for feature recognition
- **Build**: Vite with WASM plugin support

## Project Structure

```
src/                          # React frontend
  components/
    cad-viewer/               # Three.js CAD viewer (25+ files)
    part-upload/              # File upload flow
    chat/                     # AI engineering chat
    dashboard/                # Customer dashboard
    home/                     # Homepage sections
    ui/                       # shadcn/ui components
  pages/                      # Route pages
  hooks/                      # Custom React hooks
  contexts/                   # AuthContext
  stores/                     # Zustand stores
  services/                   # API services
  integrations/supabase/      # Supabase client + types

geometry-service/             # Python CAD analysis (port 5000)
  aag_pattern_engine/         # Feature recognition engine
  occwl/                      # OpenCascade wrapper
  tests/                      # Recognition accuracy tests

measurement-service/          # Python measurement (port 8081)
mesh-service/                 # Mesh tessellation
supabase/                     # Edge functions + migrations
```

## Commands

```bash
npm run dev          # Start Vite dev server
npm run build        # Production build
npm run lint         # ESLint
npm run preview      # Preview production build
```

## Code Style

- TypeScript strict mode
- Tailwind CSS for styling (no raw CSS unless necessary)
- shadcn/ui for UI components
- Zustand for client state, React Query for server state
- Supabase client via `src/integrations/supabase/`

## Spec Kit Workflow

Use slash commands for spec-driven development:

1. `/speckit.constitution` — Review/update project principles
2. `/speckit.specify` — Create feature specifications
3. `/speckit.plan` — Create implementation plans
4. `/speckit.tasks` — Generate task lists
5. `/speckit.implement` — Execute implementation
