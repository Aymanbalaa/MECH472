# Auto Offence vs Manual Defence (offence_test_v1)

Test harness for a fully **vision-only** offence (`auto_offence/program.cpp`)
against a human-driven WASD defence (`manual_defence/program.cpp`). This is
the inverse pairing of `manual_test_v3` — here the offence is automated and
you drive the defence by hand.

## Why this folder exists

To validate the offence in isolation — without a buggy or unpredictable
auto defence interfering. You move robot_B around the arena yourself and
check that the offence:

- finds you (SEARCH spin to acquire vision)
- chases you (with proportional steering and obstacle side-select)
- holds fire when an obstacle blocks line-of-sight
- fires the laser only when LOS is clear, distance is in range, and aim
  error is below the alignment tolerance

## Files

| Path | Purpose |
|---|---|
| `auto_offence/program.cpp` | mode 1, calls `wait_for_player`. Vision-only chase + LOS-gated single-shot fire. |
| `manual_defence/program.cpp` | mode 2, calls `join_player`. WASD-driven robot_B sparring partner. |

Each subfolder is a self-contained Visual Studio project (libs, BMP
assets, sln/vcxproj) — same template as `player1_offence_v2`.

## Vision template

Both processes build off prof's `mech663_assignment7` / `vision_example_6.1_centroids`
HSV pipeline:

```
copy -> lowpass_filter -> scale -> HSV threshold (s>0.20 || v<50)
     -> erode x2 -> dialate x2 -> label_image -> features() -> calculate_HSV()
```

Picked over the alternatives because:

- **Hue is brightness-invariant** — colored markers stay separable under
  lighting changes (relevant if/when we move to `vision_simulator_3.77`'s
  `example3_lighting`).
- The single threshold `s > 0.20 OR v < 50` cleanly separates colored
  markers AND black obstacles from the gray/white background in one pass.
- `features()` returns centroid + average RGB + area in one walk of the
  label image, so blob classification is a constant-time check on each
  blob (small => robot marker, large => obstacle).
- It's the prof's canonical pattern from assignment 7 — not custom code.

The defence is vision-blind by design (it's you driving), so it doesn't
run the pipeline. It only reads `S1->P[2]->x[2..3]` for the human's
telemetry print — no simulator state is used to **make decisions**.

## Vision-only purity (offence)

`player1_offence_v2` reads `S1->P[1]->x[1]` (sim heading) as a fallback
when only one of the robot's two markers is visible. The new auto_offence
removes that fallback entirely:

| Frame condition | Heading source |
|---|---|
| 2+ robot markers visible | `atan2` of marker pair, 180° resolved against direction-to-opponent |
| < 2 markers, but a pair was seen recently | hold last good vision heading |
| < 2 markers stale > 12 frames | force SEARCH state and rotate to reacquire |
| no pair ever seen yet (first frame) | aim toward seeded opponent location |

`extern robot_system *S1;` is still declared so the linker resolves
internal simulator references, but no decision logic reads it.

## Obstacles (must match across both files)

| # | (x, y) | File | Notes |
|---|---|---|---|
| 0 | **(200, 180)** | `obstacle_black.bmp` | upper-left, dark blob |
| 1 | **(440, 300)** | `obstacle_green.bmp` | lower-right, colored blob |

Diagonal layout gives the offence two distinct LOS-blocking situations
to navigate around. Two different colors so the offence's HSV obstacle
classifier (`obstacle_black` vs `obstacle_green`) is exercised — useful
when reading the first-frame label dump in the console.

## Robot starts

| Robot | (x, y, θ) | Role |
|---|---|---|
| robot_A | (500, 100, π) | auto offence, mode 1, faces left (toward defence) |
| robot_B | (140, 400, 0) | manual defence, mode 2, faces right (toward offence) |

Diagonal corners — first contact requires either robot to maneuver
around at least one obstacle.

## Manual defence controls

| Key | Action |
|---|---|
| W | forward |
| S | backward |
| A | rotate CCW (in place) |
| D | rotate CW (in place) |
| Space | fire laser (held) |
| X | exit |

The space-laser is **held**, not one-shot, so you can repeatedly try to
hit the offence back during a single test run. The offence's laser is
**one-shot** (fires once, freezes, exits).

## Build & run

Both subfolders are pre-populated with the standard simulator project
tree. Open each `.sln` in Visual Studio (v143 toolset, Debug x86 to
match the `image_transfer.lib` already in the folder), build.

## Run order

1. Start `image_view.exe` (from `project_files/image_view/`).
2. Run `auto_offence/Debug/program.exe` — press **space** to begin.
   It blocks on `wait_for_player()`.
3. Run `manual_defence/Debug/program.exe` — press **space** to begin.
   `join_player()` connects, sim starts.
4. Drive robot_B with WASD, hold space to fire back, press **X** in
   either window to quit.

## Sanity checks

- If both windows hang on start, only one process called
  `wait_for_player`/`join_player` — make sure the offence (mode 1)
  starts first.
- If the arena looks wrong on one side, confirm the obstacle BMPs and
  positions in **both** `program.cpp` files are identical.
- Only the offence (player 1) calls `view_rgb_image`. The defence
  intentionally does not, to avoid display twitching.
- The offence's first frame dumps every detected blob with HSV/area —
  use this to sanity-check the threshold if a marker isn't being seen.

## Tuning knobs (offence)

In `auto_offence/program.cpp` near the top of `main()`:

| Constant | Default | Effect |
|---|---|---|
| `FIRE_DIST` | 350 px | max range for fire eligibility |
| `STOP_DIST` | 160 px | hold-back range |
| `BACKUP_DIST` | 120 px | back-away range |
| `ALIGN_TOL` | 0.10 rad (~5.7°) | required aim accuracy to fire |
| `OBS_AVOID` | 120 px | obstacle avoidance trigger |
| `OBS_EMERG` | 40 px | emergency back-away from obstacle |
| `WALL_MARGIN` | 50 px | wall standoff |
| `LOS_BLOCK_R` | 70 px | perp-distance threshold that flags a blocked shot |
| `HEADING_STALE_THRESH` | 12 frames | how long to coast on last good vision heading before forcing SEARCH |

The hysteresis frame counts (`wall_frames=40`, `avoid_frames=60`,
`obs_engaged` widened by 50 px) prevent oscillation when the robot
hovers at the edge of a trigger zone — leave them alone unless you
see toggling.
