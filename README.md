# DRE: Direct Radar Estimation

`dre` is a ROS 2 package that brings together the full ASRL direct-radar estimation stack for online operation and real-time visualization in RViz2:

- **Dr-PoGO** (Direct Radar Pose-Graph Optimization, preprint [here](https://arxiv.org/abs/2605.04806)): combines **Direct Radar Odometry (DRO)** with **RaPlace** loop-closure detection and a **pose-graph optimizer** to produce globally consistent trajectories from FMCW radar data.
- **Dr-BA** (Direct Radar Bundle Adjustment, from the [Dr-BA](https://arxiv.org/abs/2605.07041) paper): here used live to incrementally build a persistent voxel map from DRO's local maps and pose-graph-corrected poses (no bundle-adjustment refinement online — that stays an offline step).
- **DRL** (Direct Radar Localization): localizes a live DRO stream against a previously built voxel map.

This package was formed by merging the online ROS 2 pipeline from `dr_pogo` with the mapping and localization code from the offline `dr_ba` repository, so all four estimation stages can now run live against a single vendored `ba` library. An `offline` branch of the original Dr-PoGO repo still contains the standalone version of that pipeline as used in its paper.

If you find this code useful, please consider citing our papers:

**Dr-PoGO**
```
@inproceedings{legentil2026drpogo,
  title={Dr-PoGO: Direct Radar Pose-Graph Optimization},
    author={{Le Gentil}, Cedric, Weican, Li, Brizi, Leonardo, Barfoot, Timothy D.},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```

**DRO**
```
@inproceedings{legentil2025dro,
  title={Dro: Doppler-aware direct radar odometry},
  author={{Le Gentil}, Cedric, Brizi, Leonardo, Lisus, Daniil, Qiao, Xinyuan, Grisetti, Giorgio, Barfoot, Timothy D.},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2025}
}
```

**Dr-BA**
```
@article{legentil2026drba,
  title={Dr-BA: Separable Optimization for Direct Radar Bundle Adjustment \& Localization},
  author={{Le Gentil}, Cedric and Barfoot, Timothy D.},
  journal={arXiv preprint arXiv:2605.07041},
  year={2026}
}
```

The loop-closure detection module is directly adapted from RaPlace's original code [here](https://github.com/hyesu-jang/RaPlace), so please also consider citing their work.

## Architecture overview

### Dr-PoGO pipeline (odometry, loop closure, pose graph)

| Node | Language | Role |
|------|----------|------|
| `boreas_player` | Python | Replays a Boreas sequence (radar + IMU) as ROS 2 topics |
| `dro_node` | Python | Doppler-aware direct radar odometry |
| `raplace_node` | Python | Loop-closure detection using RaPlace |
| `registration_node` | Python | Feature-based registration and direct refinement of loop-closure transformations |
| `pogo_node` | C++ | Pose-graph optimizer (Ceres) |

### Dr-BA mapping (online, no BA refinement)

| Node | Language | Role |
|------|----------|------|
| `mapping_node` | C++ | Incrementally builds an inverse-variance-weighted voxel map from DRO's local maps and keeps it corrected against pose-graph updates, saving `voxel_map.bin` and keyframe images to disk periodically |
| `map_viz_node` | Python | Renders the on-disk voxel map (whichever `/map_path` it is pointed at) as an image topic, with an optional live pose overlay |

### DRL (localization against a pre-built map)

| Node | Language | Role |
|------|----------|------|
| `loc_node` | C++ | Loads a `voxel_map.bin`, aligns each incoming DRO local map against it via direct optimization, and publishes a `DRLEstimate` |
| `initial_pose_selector` | Python | One-shot interactive tool (matplotlib click) to pick/confirm the initial pose used to seed `loc_node`, unless `skip_selector` is set |
| `loc_viz_node` | Python | Visualizes the live scan overlaid on the map plus the current pose estimate |

## Dependencies

### ROS 2 (provided by your ROS 2 installation)
- `rclpy`, `rclcpp`
- `sensor_msgs`, `nav_msgs`, `geometry_msgs`, `std_msgs`
- `message_filters`
- `cv_bridge`
- `yaml-cpp`

### C++ libraries (system)
- [Ceres Solver](http://ceres-solver.org/)
- [Eigen3](https://eigen.tuxfamily.org/)
- OpenCV
- OpenMP

These can typically be installed via your package manager (e.g., `sudo apt install libceres-dev libeigen3-dev libopencv-dev` on Ubuntu).

### C++ libraries (vendored, git submodules)
The `dre/ba` directory vendors the Dr-BA library (`voxel_map`, `local_map_scan`, and the BA/localization solvers) along with its own dependencies as submodules under `dre/ba/deps/`:
- [`lgmath`](https://github.com/utiasASRL/lgmath) — Lie-group math (SE(2)/SO(2) transformations)
- [`unordered_dense`](https://github.com/martinus/unordered_dense) — fast hash map used by the voxel map

Make sure to clone with `--recurse-submodules` (or run `git submodule update --init --recursive` after cloning).

### Python (pip)
Install all Python dependencies with:
```bash
pip install -r requirements.txt
```

The `requirements.txt` covers: `numpy`, `pandas`, `scipy`, `scikit-learn`, `scikit-image`, `opencv-python`, `matplotlib`, `PyYAML`, `pyboreas`, `torch`, `torchvision`.

## Build

First clone this repository (with submodules) into your ROS 2 workspace's `src/` directory, then build with:
```bash
cd <your_ros2_workspace>/src
git clone --recurse-submodules git@github.com:utiasASRL/dre.git
```

Then build the workspace with:
```bash
cd <your_ros2_workspace>
colcon build --packages-select dre --symlink-install
source install/setup.bash
```

`--symlink-install` is required: it installs config files as symlinks back to `src/`, so editing a config takes effect immediately without rebuilding.

## Usage

### 1. Prepare the config files

`config/config_dro.yaml` ships populated with Boreas RT defaults (also kept as a reference preset under `config/templates/`) — edit it directly, or restore it from the template if needed.
You can also customize the RaPlace, registration, pose-graph, mapping, and localization config files if desired (all parameters should have reasonable defaults though).

Note that by default, the registration node is set to use GPU if available. Depending on your hardware and the size of the local maps, this may slow down the processing of DRO's odometry (both nodes can compete for GPU resources). If you want to disable GPU usage for loop registration, simply set `use_gpu_if_available: false` in the `config/config_registration.yaml` file. It will be slower to compute the loop registration (using only 1 CPU thread), but DRO's behaviour will be more predictable.

### 2. Launch a pipeline

Several launch files are provided depending on which stages you need:

| Launch file | Nodes started | Use case |
|-------------|---------------|----------|
| `dro_launch.py` | `dro_node` + RViz | Odometry only |
| `dr_pogo_launch.py` | `dro_node`, `raplace_node`, `registration_node`, `pogo_node` + RViz | Original Dr-PoGO pipeline (odometry + loop closure + pose graph, no mapping) |
| `mapping_launch.py` / `dre_launch.py` | All of the above plus `mapping_node`, `map_viz_node` | Full online pipeline: odometry, loop closure, pose graph, and live voxel-map building. Currently identical; `dre_launch.py` is the forward-looking name |
| `drl_launch.py` | `dro_node`, `loc_node`, `initial_pose_selector`, `loc_viz_node`, `map_viz_node` + RViz | Localize live DRO output against a voxel map built in a previous run |

```bash
ros2 launch dre dre_launch.py
```

This starts the full estimation pipeline plus an RViz2 visualizer with the bundled `config/rviz_mapping.rviz` preset (each launch file uses its own matching RViz preset).

**Note:** The DRO code will attempt to leverage torch compilation if the `config/config_dro.yaml` file contains the following parameters:
```yaml
radar:
  nb_azimuths: XXX
  doppler_enabled: true/false
  resolution: Y.YYYY
```
If you chose to enable compilation, the initialization of the `dro_node` will take significantly longer (something like 30sec?). When the comilation is done and DRO is ready to process data, you should see the following message in the terminal:
```[INFO] [dro_node]: DRO ready
```
If you want to disable compilation, simply remove one of the above parameters from the config file (e.g., `resolution`).

### 3. Play a Boreas sequence

```bash
ros2 run dre boreas_player -p <path_to_sequence> -r <playback_rate>
# Example:
ros2 run dre boreas_player -p /data/boreas/boreas-2024-12-03-12-54 -r 1.0
```

You can also make it play as fast as DRO allows by setting `-r 0` (preventing to wait between messages if your hardware is fast enough to process the data faster than real-time, and allows for slower hardware to keep up by slowing down the playback rate as needed).

### 4. Localize against a previously built map (DRL)

Point `config/config_loc.yaml`'s `map_path` at a `voxel_map.bin` produced by a previous `mapping_node` run (or an offline `dr_ba` map), set an `initial_pose` (or leave `skip_selector: false` to pick one interactively when `initial_pose_selector` starts), then:
```bash
ros2 launch dre drl_launch.py
```

## Configuration

All YAML config files live under `config/`.

| File | Node | Key parameters |
|------|------|----------------|
| `config_dro.yaml` | `dro_node` | Sensor extrinsics (`T_axle_radar`), range limits, GP lengthscales |
| `config_raplace.yaml` | `raplace_node` | `min_time_diff`, `max_odom_drift`, `max_img_size` |
| `config_registration.yaml` | `registration_node` | `lowe_ratio`, `ransac_thr`, `max_img_size`, `use_gpu_if_available` |
| `config_pogo.yaml` | `pogo_node` | Odometry/loop noise std-devs, loss scales, `estimate_bias` |
| `config_mapping.yaml` | `mapping_node` | `voxel_map_resolution`, `max_dist`, keyframe gate (`max_kf_dist`, `max_kf_rot`), intensity noise model, worker-thread throttling (`max_moved_per_update`, `min_moved_per_update`) |
| `config_loc.yaml` | `loc_node`, `initial_pose_selector`, `loc_viz_node` | `map_path`, `initial_pose`, scan-blurring and optimization parameters |

## Custom messages

| Message | Fields |
|---------|--------|
| `RadarInfo` | Radar scan metadata (timestamps, frequency, etc.) |
| `LocalMapInfo` | Accumulated local map header, resolution, 2-D pose (`x`, `y`, `theta`) |
| `LoopCandidate` | Query/candidate timestamps, match score, image paths, local map resolution |
| `DRLEstimate` | Localization pose (`x`, `y`, `theta`) with solver `iterations` and final `cost` |

## Output

Atop ROS2 topics shown in RViz, the pipeline outputs the following to an output directory specified in the launch file (default is in the install space under `<ros2_ws>/install/dre/share/dre/<sequence_id>/`):
- `odometry_result/<sequence_id>.txt`: DRO odometry trajectory using the Boreas format.
- `pose_graph_traj.txt`: Pose-graph optimized trajectory in with `timestamp(us) x y theta` format.
- `voxel_map.bin`, `local_maps/`, `cumulated_returns/`: `mapping_node`'s persistent voxel map and the keyframe images backing it (under the `map_output_path` argument of `mapping_launch.py`/`dre_launch.py`, default `output/boreas-live/`).

### TODOs

- [ ] Making the dense loop registration refinement faster !! (need to profile first)
- [ ] Improving documentation
- [ ] Looking at making DRO even faster? (real time on my RTX 500 Mobile GPU (30W), so not a priority)
- [ ] Adding the 3D odometry output as for 3DRO
- [ ] Reconciling `dre_launch.py` and `mapping_launch.py` into a single forward-looking entry point
