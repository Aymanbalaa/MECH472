# MECH 472/663 — Robot Laser Wars

Course project files and lecture dumps for MECH 472/663 (mobile robotics + computer
vision). The project is a two-player "laser wars" simulation: one robot plays
offence (chase and shoot) and the other plays defence (evade and hide). All
perception is done through computer vision on a simulated camera feed — no robot
state is read from the simulator except heading as a fallback.

This repo contains several iterations of each player's controller, plus the
shared simulator/vision libraries, image assets, and a standalone image viewer.

---

## Repository Layout

```
MECH472/
├── README.md
├── .gitignore
└── project_files/
    ├── Debug/                    # prebuilt binaries / objs
    ├── image_view/               # standalone image viewer utility
    ├── ayman_old/                # legacy code
    │   ├── auto_offence/
    │   └── robot_B_player2/
    ├── naim_defense/             # teammate's defence reference
    ├── defence_manual_test/      # manual-control defence for testing
    ├── player1_offence/          # offence controller (v1)
    ├── player1_offence_v2/       # offence controller (v2 — current)
    ├── player2_defence/          # defence controller (v1)
    ├── player2_defence_v2/       # defence controller (v2)
    └── player2_defence_v3/       # defence controller (v3 — current)
```

Each player directory is a self-contained Visual Studio project (`program.sln`,
`program.vcxproj`) with its own copy of the simulator libraries, headers, and
image assets so it can be built and run independently.

---

## The Simulator

The arena is a 640×480 image with:
- **robot_A** — player 1's robot (controlled in `mode = 1`)
- **robot_B** — player 2's robot (controlled in `mode = 2`)
- **2 obstacles** at `(320, 220)` — `obstacle_black.bmp` and `obstacle_green.bmp`
- **background** — `background.bmp`

Geometry constants (must match between both players):

| Constant   | Value | Meaning |
|------------|-------|---------|
| `D`        | 121.0 | wheel base (px) |
| `Lx, Ly`   | 31, 0 | laser offset in robot frame |
| `Ax, Ay`   | 37, 0 | axis-of-rotation offset |
| `alpha_max`| π/2   | laser/gripper sweep range |
| `max_speed`| 120   | px/s |

### Two-Player Setup

Both programs call `activate_simulation(...)` with identical parameters, then:

- Player 1 calls `set_simulation_mode(1)` and `wait_for_player()`
- Player 2 calls `set_simulation_mode(2)` and `join_player()`

The simulator synchronises the two processes via shared memory
(`shared_memory.cpp`). Inputs are supplied as RC-style pulse widths
(`pw_l`, `pw_r`, `pw_laser` in 1000–2000 µs, neutral 1500).

### Game Rules

- **Offence**: one laser shot per turn — no sweeping (laser angle locked while
  firing).
- Robots cannot leave the screen (turn forfeit).
- Hitting an obstacle deducts marks.
- Obstacles block the laser — defence can use them as cover.

---

## Vision Pipeline

Both players use the same HSV-based pipeline (adapted from the professor's
assignment 7 / `vision_example_6` / `simulator_example2`):

```
acquire_image_sim → copy → lowpass_filter → scale
  → HSV threshold (s > 0.20  OR  v < 50)
  → erode × 2 → dialate × 2
  → label_image → detect_objects (centroid + features)
```

### Blob Classification

`detect_objects()` walks every connected component and classifies it by area
and HSV:

| Area               | Class    | Assignment |
|--------------------|----------|------------|
| `< 200`            | rejected | noise |
| `200 – 2500`       | robot marker | nearer of own/opponent previous centre |
| `> 2500`           | obstacle | colour bucketed by hue (black / red / orange / green / blue) |

The centre of the own robot and the opponent are tracked across frames using
**proximity to previous-frame centroids**, so two markers belonging to the same
robot are aggregated correctly.

### Heading Estimation

When two robot markers are detected, heading is `atan2` of the vector between
them. The 180° ambiguity is resolved by comparing both candidates against a
direction prior:

- **Offence**: direction *toward* the opponent.
- **Defence**: direction *away* from the opponent (proxy), then refined to
  the potential-field desired heading.

If only one marker is visible, heading falls back to `S1->P[i]->x[1]` (sim
state). This is the only simulator state read.

---

## Controllers

### Player 1 — Offence (`player1_offence_v2/`)

State machine: **SEARCH → CHASE → FIRE**

| Constant      | Value | Purpose |
|---------------|-------|---------|
| `FIRE_DIST`   | 350   | max distance to take a shot |
| `STOP_DIST`   | 160   | close-range shot trigger |
| `BACKUP_DIST` | 120   | back away to re-aim |
| `ALIGN_TOL`   | 0.10 rad | heading error to fire |
| `OBS_AVOID`   | 120   | obstacle avoidance trigger |
| `OBS_EMERG`   | 40    | emergency back-off zone |
| `WALL_MARGIN` | 50 px | wall hysteresis box |
| `LOST_THRESH` | 25 frames | back to SEARCH when opponent missing |

Drive priority (highest first):

1. **Wall** — steer back toward arena centre with hysteresis (40-frame hold).
2. **Obstacle** — cross-product side selection picks left/right detour;
   emergency mode reverses if too close and facing the obstacle.
3. **State machine** — SEARCH spins CCW, CHASE drives with proportional
   steering, FIRE fine-tunes aim then shoots once.

A line-of-sight check rejects shots blocked by an obstacle (perpendicular
distance from segment to obstacle < 70 px ⇒ blocked).

The laser fires exactly once: when aim error < `ALIGN_TOL`, `clear_shot`,
`dist < FIRE_DIST`, and `t > 3 s`. Then the program freezes the frame, draws
overlays, and exits.

### Player 2 — Defence (`player2_defence_v3/`)

Strategy: compute a **hiding spot** behind the best obstacle, navigate there
with a **potential field**, stop once the line of sight to the opponent is
broken.

Hiding spot per obstacle:
```
hide = obstacle + hide_dist * (obstacle - opponent) / |obstacle - opponent|
```
with `hide_dist = 110`. The chosen spot maximises `d_opp − 0.3 · d_robot`
(far from opponent, not too far from us). A persistent obstacle memory
(`known_obs_*`) keeps last-known positions when vision temporarily loses an
obstacle (e.g. a robot covers it).

Potential field:

| Force         | Gain  | Form |
|---------------|-------|------|
| Attractive    | `Ka = 4.0` | unit vector toward hide spot |
| Obstacle      | `Kr = 15000` | `1/r²` repulsion from each obstacle |
| Boundary      | `Kb = 15000` | `1/r²` repulsion from each wall |

The desired heading is `atan2(Fy, Fx)`. A two-phase waypoint navigator
(`wp_state` 0=rotate, 1=drive) turns in place until aligned, then drives with
proportional steering. When line of sight is broken (`in_los == false`) or
within `arrive_dist = 30 px` of the waypoint, the robot stops.

### Other Controllers

- `player1_offence/` — earlier offence (RGB threshold + `centroid2` instead of
  HSV + `features`).
- `player2_defence/` and `_v2/` — earlier defence iterations
  (state machine SEARCH → FLEE → STRAFE → HIDE in v1, potential field in v2).
- `defence_manual_test/` — defence with WASD manual control for opponent
  (used for tuning).
- `naim_defense/` — teammate Naim's defence implementation, kept as reference.
- `ayman_old/auto_offence/` and `ayman_old/robot_B_player2/` — early
  experiments.

---

## Shared Components

Each project ships a copy of:

| File | Purpose |
|------|---------|
| `vision.h / .cpp` | image ops: `copy`, `scale`, `lowpass_filter`, `erode`, `dialate`, `label_image`, `centroid`, `draw_point_rgb` |
| `vision_simulation.h / .cpp` | `robot_system` class, `activate_simulation`, `acquire_image_sim`, `set_inputs`, mode/position helpers, `wait_for_player` / `join_player` |
| `robot.h / .cpp` | `robot` class — state vector, `sim_step`, `set_inputs`, `calculate_outputs` |
| `update_simulation.h / .cpp` | hooks to update obstacles / background per frame |
| `shared_memory.h / .cpp` | `CreateFileMapping` wrapper for inter-process sync |
| `timer.h / .cpp` | `high_resolution_time` wall-clock helper |
| `image_transfer.h` + `.lib` | low-level image acquisition (Windows DirectShow) |
| `atlsnd.lib`, `strmbasd.lib` | DirectShow base-class libs |

Image assets: `robot_A.bmp`, `robot_B.bmp`, `background.bmp`, `obstacle_*.bmp`,
plus `output.bmp` for offline processing.

---

## Building & Running

**Requirements:** Windows + Visual Studio (C++ desktop workload). The projects
target the v143 toolset and use the included DirectShow `.lib` files — they
will not build on Linux/macOS without porting.

1. Open `program.sln` in the desired controller folder
   (e.g. `player1_offence_v2/program.sln`).
2. Build in `Debug` (the prebuilt `Debug/program.exe` is also included).
3. To run a two-player match, start **player 1** first (it calls
   `wait_for_player`), then start **player 2** in a second console (it calls
   `join_player`). Both processes attach to the same shared-memory simulator.
4. Press **Space** at the prompt to begin. Press **X** at any time to exit.

**`image_view.exe`** (in `project_files/image_view/`) renders the simulator
output. On high-DPI monitors, set
*Properties → Compatibility → Disable display scaling on high DPI settings*
before launching, per `image_view_readme.txt`.

---

## Versioning Notes

The `*_v2` / `*_v3` directories are full-tree forks rather than diffs — each
contains its own copy of the libs and assets. Rough timeline:

- **player1_offence → player1_offence_v2**: switched from RGB threshold +
  `centroid2` to HSV threshold + `features`; added obstacle/wall hysteresis,
  cross-product side selection, and explicit FIRE state.
- **player2_defence → _v2**: replaced state machine with potential-field
  navigation.
- **_v2 → _v3**: added persistent obstacle memory, refined heading
  ambiguity resolution using the desired heading from the potential field.

---

## License / Attribution

Course material for MECH 472/663. Vision pipeline derived from the
professor's `vision_example_6`, `simulator_example2`, and assignment 7. Kept
here as a "dump for lectures and project files for robot laser wars".
