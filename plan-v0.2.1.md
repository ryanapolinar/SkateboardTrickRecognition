# Plan v0.2.1 — Web HCI & flatground scope (patch)

> **Patch on top of `plan-v0.2.md` (the base).** Nothing in v0.2 is undone unless this
> file says otherwise. Sections this patch **overrides** in v0.2: §1 scope (adds the
> flatground mandate), §4 step 5 (web → CLI), §8 UX (web-first), §10 M4 (Gradio line),
> §12 item on the Gradio add-on. Everything else stands.

---

## 1. Why this patch (two decisions)

1. **HCI = an HTML webpage is the primary interface**, not the CLI. Single local page,
   drag a clip, read the trick. The CLI stays only for batch/automation.
2. **Flatground-only is a hard scope guarantee.** Every class we try to detect today is
   flatground; anything that needs an obstacle or a ramp is out, and a test enforces it.

---

## 2. HCI — local web app is v1 default

```
skateid serve         # -> opens http://127.0.0.1:8000 in the default browser
```

- **Frontend:** one static `index.html` written by hand (plain HTML/CSS/JS — no node.js,
  no build step, no framework). It has: drag-and-drop + a "choose file" button, an inline
  `<video>` preview, one **"Identify trick"** button, a result area showing the trick name
  big + a confidence bar (or a clear **"not sure"** state), an optional **"show annotated"**
  toggle that plays the overlay clip, and a "try another" reset.
- **Backend:** FastAPI + uvicorn in `app.py`, three routes total:
  | Route | Behaviour |
  |---|---|
  | `GET  /` | serves `index.html` |
  | `POST /predict` | takes the uploaded clip (multipart) → runs the recognizer → returns `{trick, confidence, top3, overlay_url}` |
  | `GET  /overlay/<id>.mp4` | serves the annotated clip for playback |
- **The core is untouched:** `recognize.py` is the single source of truth. The web app is a
  thin ~80-line wrapper; the CLI and the page call the **same** pipeline. No second model,
  no divergence.
- Runs localhost, offline, no accounts, no telemetry.

**Why FastAPI (replacing v0.2's "optional Gradio M4"):** you asked for a real HTML page.
FastAPI + uvicorn serves a hand-written `index.html` plus file-upload/JSON with zero JS
toolchain. (Flask is an acceptable alternative if you prefer it — pick one; default is
FastAPI.) Gradio is dropped from the plan.

---

## 3. Flatground-only scope (mandatory)

Adds this rule to v0.2 §1:

> **v1 recognizes FLATGROUND tricks only** — no grinds/slides on obstacles, no ramp/vert/aerial.

**Included (all flatground):** ollie, kickflip, heelflip, shuvit & pop-shuvit, 360 flip /
tre-flip, bigspin, frontside/backside 180 and 540, plus double/triple variants produced by
the four components. Every head (flip, board spin, body spin, stance) is a rotation that can
happen on flat ground.

**Explicitly excluded from the v1 label set:**
- Grinds & slides (rail, ledge, curb) — obstacle tricks.
- Vertical / ramp / vert / half-pipe aerials.
- Manuals & nose-manuals — flatground, but a *balance-hold*, not a rotation; they don't fit
  the four-component model. Deferred.
- Casper / anti-casper and other balance tricks — deferred.
- Any trick whose name or component combination needs an obstacle.

**Guardrail so it stays true (not just a promise):**
- `data/flatground_allowlist.csv` — the approved labels + the allowed component combinations.
- `tests/test_scope.py` — fails CI if any `manifest.csv` row has a label (or component combo)
  that is not on the flatground allowlist.
- Manifest build rejects non-flatground rows at ingestion.
- `tricks.json` dictionary is restricted to the flatground names (SkateAI's 58 entries are all
  flatground, but we drop any that need an obstacle).
- README capture guide: film only flatground attempts; no obstacle in frame.

**Data alignment check — all three sources are already flatground, so no new data is needed:**
- SkateboardML → ollie + kickflip ✓
- SkateAI → BATB flatground battles (kickflips, shuvits, bigspins, tre-flips, 180/360/540) ✓
- Swinburne → BATB flatground, vocabulary-only ✓

---

## 4. Files changed vs v0.2

| File | Change |
|---|---|
| `skateid/app.py` (NEW) | FastAPI server: `/`, `/predict`, `/overlay` |
| `skateid/web/index.html` (NEW) | the single static page (drag-drop, preview, result, annotate) |
| `data/flatground_allowlist.csv` (NEW) | flatground labels + component combos |
| `tests/test_scope.py` (NEW) | CI guardrail: rejects any non-flatground label/combo |
| `pyproject.toml` (EDIT) | add `fastapi`, `uvicorn`, `python-multipart` |
| `skateid/cli.py` (EDIT) | add `serve` subcommand |
| `recognize.py / features.py / model.py / pose.py / board.py / train.py / eval.py` | **unchanged** — web UI reuses the pipeline |

---

## 5. Milestone delta (M3 / M4)

- **M3 now includes:** `skateid serve` working end-to-end on the 2-class model — open
  `http://127.0.0.1:8000`, drop a clip, read "kickflip 0.87" (or **"not sure"**) in <2 s,
  with the annotated overlay playable. CLI `--batch`/`--json` still available for automation.
- **M4 now includes:** web polish (annotated-overlay toggle, top-3 list, drag-drop niceties,
  an in-page batch list). The ONNX-export / speed / board-distill work from v0.2 M4 is unchanged.

---

## 6. Acceptance for the HCI

- `skateid serve` → `http://127.0.0.1:8000` → drop a clip → trick + confidence (or "not sure")
  within ~2 s, using only the browser. No console needed for normal use.
- One recognizer core used identically by the web page and the CLI.
- Flatground-only enforced: the repo cannot be trained/have CI pass with a non-flatground
  label or component combo in the manifest.