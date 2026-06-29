# rm75_control GitHub Notes

Repository root: `/media/camp/EXT_DRIVE/rm75_control`  
Remote: `https://github.com/NingMeng0903/rm75_control.git`

## Active layout

- `rm75_control/`: installable Python package (session, motion, force scan, backend)
- `configs/`: robot YAML defaults (`rm75f_default.yaml`)
- `apps/`: application entry points
- `tmp/`: bench / demo scripts (force scan, CANFD sin, recovery)
- `tests/`: offline tests
- `MD/`: operator and Git notes

External runtime dependency (not in this repo):

- [RM_API2](https://github.com/RealManRobot/RM_API2) — set `RM_API2_PYTHON` in `env.sh`

## First push scope

Commit:

- `rm75_control/`
- `configs/`
- `apps/`
- `tests/`
- `MD/`
- `README.md`
- `env.sh`, `requirements.txt`, `setup.py`, `pyproject.toml`

Do not commit:

- local conda / venv trees
- `*.egg-info/` build artifacts
- machine-specific secrets or IPs if you prefer to keep them local
- large logs or captured data under `tmp/` unless explicitly needed

## Git bootstrap

First time on a new machine:

```bash
cd /media/camp/EXT_DRIVE/rm75_control
git init
git add rm75_control/ configs/ apps/ tests/ MD/ README.md env.sh requirements.txt setup.py pyproject.toml
git status
git commit -m "Initial commit: RM75 control wrapper and force-scan demos."
git branch -M main
git remote add origin https://github.com/NingMeng0903/rm75_control.git
git push -u origin main
```

## Routine upload (after local changes)

```bash
cd /media/camp/EXT_DRIVE/rm75_control
git status
git add .
git commit -m "大修改，解决4个问题，没有测试"
git push origin main
```

Upload everything that changed (review `git status` first):

```bash
cd /media/camp/EXT_DRIVE/rm75_control
git add -u
git add MD/
git status
git commit -m "Describe your change here."
git push origin main
```

If the remote is ahead:

```bash
cd /media/camp/EXT_DRIVE/rm75_control
git pull origin main --rebase
git push origin main
```

## Current command entry points

Activate env before any script:

```bash
source /media/camp/EXT_DRIVE/rm75_control/env.sh
```

### Force-position scan (tool Y sin + tool Fz)

```bash
source /media/camp/EXT_DRIVE/rm75_control/env.sh
python /media/camp/EXT_DRIVE/rm75_control/tmp/tcp_z_spring.py \
  --prepress --trajectory sin_tool_y --z-force 3
```

### Cartesian CANFD sin (position only, lag benchmark)

```bash
source /media/camp/EXT_DRIVE/rm75_control/env.sh
python /media/camp/EXT_DRIVE/rm75_control/tmp/sin_y_cartesian.py --no-ruckig
```

### Recover after planned/stream force conflict

```bash
source /media/camp/EXT_DRIVE/rm75_control/env.sh
python /media/camp/EXT_DRIVE/rm75_control/tmp/recover_force_stream.py
```

## Notes

- `README.md` is the primary GitHub landing page.
- RM_API2 stays outside this repo; document its path in `env.sh` on each machine.
- Do not force-push `main` unless you intentionally rewrite remote history.
