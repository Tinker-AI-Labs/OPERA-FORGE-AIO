# Ideas / UI Reference — PUBLISHA & GAMEA

Reference notes from early UI concept mockups, captured for the repo so the direction isn't lost between build sessions.

## PUBLISHA ("LEXICON: Author Hub" concept)

**Purpose:** Indie-author operations hub — manuscript management, sales/marketing, and analytics in one dashboard.

**Core layout:**
- Top nav: Dashboard / My Books / Writing Projects / Marketing / Analytics / Account
- **Writers Profile panel** — avatar, short bio, genres, active project count
- **My Books list** — status (Published/Drafting/Outline) with per-title quick actions: Edit, Sales Data, Marketing
- **Writing Projects panel:**
  - Active manuscript card: title, chapter count, word count, % progress, progress bar, buttons for Open Editor / Manuscript Settings / Export
  - Secondary project cards for other in-progress works (e.g. non-fiction idea in Drafting, short story collection in Outline)
- **Recent Analytics section** (date-range filterable):
  - Book Sales this month (bar chart)
  - Royalties Overview (line chart, trend over time)
  - Reader Engagement (pie chart breakdown)

**Notes:** Two parallel free-AI-account builds (PUBLISHA + GAMEA). No backend/pipeline decisions made yet for PUBLISHA — this is UI/UX direction only.

---

## GAMEA ("PixelSmith" concept) — OPERA-GAMEA

**Purpose:** Local 3D game-asset generation front end, sitting on top of the local pipeline (ComfyUI + Hunyuan3D-2 + TripoSR + SDXL, Meshroom for photogrammetry, Blender for mesh cleanup — all on RX 6600/ROCm).

**Core layout:**
- Top bar: Project selector, user, New Asset / Generate / Export (FBX/OBJ/GLTF) / Settings
- **Left panel — Project/Library:** folder tree (Models, Textures, Audio, Icons, UI) with thumbnails of generated assets
- **Center — Canvas:**
  - Prompt box (e.g. "stylized fantasy knight, full plate armor, sword, shield, dynamic pose, low-poly, hand-painted texture, unreal engine")
  - Live 3D preview viewport of the generated model
  - **Generation Workflow controls:** Model Type, Style, Texture Res, Materials, Complexity slider, Roughness slider, Rigging (e.g. auto-rig)
  - Generate Asset button
- **Right panel — Asset Preview & Details:**
  - Thumbnail variants (diffuse/normal/roughness maps, alt angles)
  - Stats: polygon count, tri count, bone count, texture map list
  - Asset Queue — shows in-progress/queued generations

**Backend mapping (already in progress):**
- ComfyUI — orchestration
- Hunyuan3D-2 — shape generation (confirmed working end-to-end on RX 6600 as of 8/6; texture stage still untested)
- TripoSR / SDXL — supplementary generation
- Meshroom — photogrammetry input (installing)
- Blender — mesh cleanup bridge
- Tripo.ai bridge — still needed to connect Tripo-side generation into the pipeline

Lives in the OPERA-FORGE-AIO repo, renamed OPERA-GAMEA in-UI (FORGE was already taken by T1NK3R.FORGE).

---

*Status: both are early visual-direction mockups, not locked specs. Feature sets and layout still open to iteration.*
