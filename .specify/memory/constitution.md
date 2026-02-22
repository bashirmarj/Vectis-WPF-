<!--
  Sync Impact Report
  ===================
  Version change: 0.0.0 (template) -> 1.0.0 (initial ratification)
  Modified principles: All new (template placeholders replaced)
  Added sections:
    - I. RFQ-First Product
    - II. Reliable CAD Viewer
    - III. Feature Recognition Accuracy
    - IV. Security Gate
    - V. Industrial UX
    - VI. Performance
    - Scope Rules
    - Testing Rules
    - Security & Privacy Rules
    - Repo Hygiene
  Removed sections: None (template placeholders replaced)
  Templates requiring updates:
    - .specify/templates/plan-template.md - no changes needed (Constitution Check section is generic)
    - .specify/templates/spec-template.md - no changes needed (user story structure aligns)
    - .specify/templates/tasks-template.md - no changes needed (phase structure aligns)
  Follow-up TODOs: None
-->

# Vectis Machining RFQ Platform Constitution

## Core Principles

### I. RFQ-First Product

The website MUST function primarily as an RFQ intake product. Every design decision
MUST optimize for "successful RFQ submission" above all other goals. The core flow is:

1. User signs in (security gate)
2. User uploads CAD pack (STEP + DXF + PDF, optionally zipped)
3. In-browser CAD viewer renders the uploaded files
4. Feature recognition produces a feature tree with target accuracy
5. User reviews and submits the RFQ
6. System delivers the RFQ to belmarj@vectismanufacturing.com
7. System tracks the submission event via analytics

Existing site pages and content structure MUST be preserved unless changes are
directly required to support the RFQ + viewer + recognition flow.

### II. Reliable CAD Viewer

The CAD viewer MUST work reliably on modern browsers (Chrome, Firefox, Edge, Safari)
and typical corporate laptops. Failure modes (unsupported files, huge assemblies,
corrupted files) MUST be handled gracefully with clear, user-facing UX messages.
The viewer MUST NOT silently fail.

### III. Feature Recognition Accuracy

Feature recognition MUST target a minimum of 80% accuracy on a defined, repeatable
evaluation test set. Accuracy MUST be measured consistently across releases. When
recognition confidence is low, the UI MUST communicate uncertainty to the user
rather than presenting uncertain results as definitive.

### IV. Security Gate

Users MUST sign in before uploading any files. Sign-in is the primary gate against
spam and abuse. No file upload, viewer, or recognition functionality is accessible
to unauthenticated users.

### V. Industrial UX

The UI MUST maintain an industrial/professional tone: clean, minimal, and credible.
Gimmicky styling is prohibited. UX is the top quality priority when trading off
between competing concerns.

### VI. Performance

Bundle size and client JS load time MUST be optimized, particularly because of the
CAD viewer's weight. Heavy unnecessary client-side JavaScript MUST be avoided.
Performance is critical for viewer usability on corporate networks and hardware.

## Scope Rules

### Near-Term Focus (Current Priority)

All development effort MUST focus on these three areas unless a task explicitly
unblocks them:

1. **CAD viewer functionality** - reliable rendering and interaction
2. **Feature recognition** - accurate feature tree with measurable 80%+ accuracy
3. **RFQ submission flow** - upload, review, submit, email delivery

### Explicitly Deprioritized

The following areas MUST NOT receive development effort unless they directly
unblock the near-term focus:

- RFQ completeness gating or forcing extra fields
- Extensive RFQ data capture requirements
- Full end-to-end quoting automation

### Price Positioning

The site MUST support "fast, competitive pricing" messaging. It MUST NOT promise
exact pricing or lead times unless explicit policies are added later.

## Testing Rules

### Viewer Testing

- A representative STEP file test pack MUST be maintained and re-tested every release.
- Common failure modes (unsupported files, huge assemblies, corrupted files) MUST be
  tested with clear UX verification.

### Feature Recognition Testing

- A repeatable evaluation set (N sample parts) MUST be maintained.
- Accuracy MUST be measured consistently; acceptance target is 80% or higher.
- Results MUST be tracked across releases.

### RFQ Flow Testing

- Sign-in requirement MUST be verified before upload access.
- Multiple file upload (STEP/DXF/PDF) MUST be tested.
- Submission MUST trigger email delivery to belmarj@vectismanufacturing.com.
- RFQ submission analytics event MUST fire.

### Regression Checks (Every Release)

Every release MUST verify:

1. Sign-in flow
2. File upload
3. CAD viewer rendering
4. Feature recognition output
5. RFQ submission
6. Confirmation UX
7. Email delivery

## Security & Privacy Rules

### Secrets Management

- API keys, email credentials, and storage tokens MUST NOT appear in client code.
- `.env*` files, credentials, and tokens MUST NOT be committed to the repository.

### Data Confidentiality

- All uploaded CAD files and RFQ data MUST be treated as confidential by default.

### Upload Handling

- Server-side validation of file types and size limits is REQUIRED.
- All upload errors MUST produce clear, user-facing error messages. Silent failures
  are prohibited.

### Storage

- Current approach: email delivery is acceptable short-term.
- Future: migrate uploads to Google Drive (or equivalent) using least-privilege
  access and secure links.

### Abuse Protection

- Sign-in is the primary gate. Rate limiting and basic abuse protections SHOULD be
  added if abuse patterns emerge.

## Repo Hygiene

### Gitignore Rules (Vite / React)

The following MUST be excluded from version control:

- **Secrets**: `.env*`, credentials, tokens
- **Local artifacts**: `*.log`, `*.tmp`, `*.zip`, `/uploads`, `/storage`, `/backups`
- **CAD files**: `*.step`, `*.stp`, `*.iges`, `*.igs`, `*.stl`, `*.dwg`, `*.dxf`, `*.pdf`

### Documentation

The README MUST document deployment configuration including:

- Authentication setup
- Email routing
- Upload/storage approach
- Analytics configuration

## Governance

This constitution supersedes all other development practices for the Vectis Machining
website. Amendments require:

1. Documentation of the change and rationale
2. Review and approval by the project owner
3. Version increment following semantic versioning

If a request conflicts with the near-term focus (viewer + recognition + RFQ submit),
it MUST be deferred unless it unblocks those goals.

**Version**: 1.0.0 | **Ratified**: 2026-02-18 | **Last Amended**: 2026-02-18
