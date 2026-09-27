# SkateID - Flatground Skateboard Trick Recognition

SkateID predicts flatground skateboard trick names with confidence (or abstains with "not sure") from video clips.

## Features
- Continuous 3-axis rotation prediction (`flip`, `board_spin`, `body_spin`)
- Dictionary lookup over canonical tricks
- Stance toggle (`auto` / `regular` / `goofy`, default `auto`)
- Web UI (FastAPI + single vanilla `index.html`) and CLI
