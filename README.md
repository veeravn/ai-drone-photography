# AI-Drone-Photography

AI-Photographer is a modular toolset for automated next-best-view (NBV) photography using a drone / gimbal. It contains NBV planning, a simple agent to fly and capture, deduplication using pose + perceptual hash, and helper I/O modules for cameras and rangefinders.

## Quick highlights
- NBV planning: sample candidate views and score next-best views.
- Agent: autonomous capture loop that avoids duplicates and logs kept shots.
- Dedupe: hybrid spatial + perceptual-hash duplicate detection.
- Dry-run friendly: Webcam / simulated hardware classes let you run without a drone.

## Requirements
Install dependencies (macOS / Linux):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Consider pinning versions for reproducible installs.

## Configuration
Edit `config.yaml` at the repo root. Many scripts default to `config.yaml` in the repo root; runner scripts accept `--config` to point elsewhere.

## Common commands
- Run the NBV agent:
```bash
python scripts/run_agent.py --config config.yaml
```
- Demo fly-and-shoot (dry mode):
```bash
python scripts/fly_and_shoot.py --config config.yaml --shots 10 --dry
```
- Run the unit tests (if present):
```bash
# add tests/ and run via pytest
pytest
```

## Important notes for developers
- CSV logging helper for ShotMeta was renamed to `append_shotmeta_csv` in `ai_photographer.dedupe`. Update any custom code to call that name.
- `compute_phash` in `ai_photographer.dedupe` resizes images to a conservative target size to support larger `phash_size` and `highfreq_factor`. You can tune `phash_size` and `highfreq_factor` in code or via configuration to trade off hash resolution vs. compute cost.
- Scripts were updated to accept `--config` (defaulting to the repository `config.yaml`).

## Project layout
- ai_photographer/
  - agent.py — NBVAgent, main agent loop
  - nbv_planner.py — sample views, scoring and selection
  - dedupe.py — ShotMeta, ShotDB, duplicate decision, CSV logging (`append_shotmeta_csv`)
  - io_utils.py — Webcam, image save, CSV helpers (generic)
  - gimbal_control.py — gimbal helpers
  - rangefinder_reader.py — range sensor reader
- scripts/
  - run_agent.py — main runner (async)
  - fly_and_shoot.py — simpler demo script
- config.yaml — default configuration
- requirements.txt — Python deps

## Contributing
- Run linters and tests before opening a PR.
- Keep changes small and document behavioral changes (e.g., CSV helper rename).
- For hardware interactions, add dry-mode coverage.

If you want, I can:
- Add a small CI job or tests for the dedupe logic.
- Provide a pinned `requirements.txt` that is known-good for macOS.

```# filepath: /Users/veerav/Documents/GitHub/ai-photographer/README.md
# ai-photographer

AI-Photographer is a modular toolset for automated next-best-view (NBV) photography using a drone / gimbal. It contains NBV planning, a simple agent to fly and capture, deduplication using pose + perceptual hash, and helper I/O modules for cameras and rangefinders.

## Quick highlights
- NBV planning: sample candidate views and score next-best views.
- Agent: autonomous capture loop that avoids duplicates and logs kept shots.
- Dedupe: hybrid spatial + perceptual-hash duplicate detection.
- Dry-run friendly: Webcam / simulated hardware classes let you run without a drone.

## Requirements
Install dependencies (macOS / Linux):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Consider pinning versions for reproducible installs.

## Configuration
Edit `config.yaml` at the repo root. Many scripts default to `config.yaml` in the repo root; runner scripts accept `--config` to point elsewhere.

## Common commands
- Run the NBV agent:
```bash
python scripts/run_agent.py --config config.yaml
```
- Demo fly-and-shoot (dry mode):
```bash
python scripts/fly_and_shoot.py --config config.yaml --shots 10 --dry
```
- Run the unit tests (if present):
```bash
# add tests/ and run via pytest
pytest
```

## Important notes for developers
- CSV logging helper for ShotMeta was renamed to `append_shotmeta_csv` in `ai_photographer.dedupe`. Update any custom code to call that name.
- `compute_phash` in `ai_photographer.dedupe` resizes images to a conservative target size to support larger `phash_size` and `highfreq_factor`. You can tune `phash_size` and `highfreq_factor` in code or via configuration to trade off hash resolution vs. compute cost.
- Scripts were updated to accept `--config` (defaulting to the repository `config.yaml`).

## Project layout
- ai_photographer/
  - agent.py — NBVAgent, main agent loop
  - nbv_planner.py — sample views, scoring and selection
  - dedupe.py — ShotMeta, ShotDB, duplicate decision, CSV logging (`append_shotmeta_csv`)
  - io_utils.py — Webcam, image save, CSV helpers (generic)
  - gimbal_control.py — gimbal helpers
  - rangefinder_reader.py — range sensor reader
- scripts/
  - run_agent.py — main runner (async)
  - fly_and_shoot.py — simpler demo script
- config.yaml — default configuration
- requirements.txt — Python deps

## Contributing
- Run linters and tests before opening a PR.
- Keep changes small and document behavioral changes (e.g., CSV helper rename).
- For hardware interactions, add dry-mode coverage.

If you want, I can:
- Add a small CI job or tests for the dedupe logic.
- Provide a pinned `requirements.txt` that is