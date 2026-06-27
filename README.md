# rm75_control

RealMan RM75-F integrated control wrapper: session management, force-position streaming, and tool-frame scan demos.

## Dependency layout

This repo contains **only** the `rm75_control` Python package and demo scripts. It does **not** vendor the official RealMan SDK.

| Component | Location | In this repo? |
|-----------|----------|---------------|
| `rm75_control` package | this repo | yes |
| Official SDK (`Robotic_Arm`, `rm_robot_interface.py`, …) | [RM_API2](https://github.com/RealManRobot/RM_API2) (clone separately) | no — runtime via `PYTHONPATH` |

Application code imports the local package, for example:

```python
from rm75_control import RobotSession
```

The backend (`rm75_control/backend/realman.py`) imports `Robotic_Arm` from RM_API2 at runtime.

## Setup

1. Clone [RM_API2](https://github.com/RealManRobot/RM_API2) and place it on disk (e.g. `/path/to/RM_API2`).
2. Edit `env.sh` — set `RM75_CONTROL_ROOT`, `RM_API2_PYTHON`, and conda env path.
3. Install Python deps and activate:

```bash
source env.sh
pip install -r requirements.txt
pip install -e .
```

## Quick run (force scan demo)

```bash
source env.sh
python tmp/tcp_z_spring.py --prepress --trajectory sin_tool_y --z-force 3
```

## Project layout

```
rm75_control/     # installable package (core, backend, motion, force, …)
configs/          # robot YAML defaults
apps/             # application entry points
tmp/              # experimental / bench scripts
tests/            # offline tests
```
