# 🚨 Deprecation Notice

This repository is no longer maintained.

Since keeping this repo up-to-date with the continuous updates in Isaac Lab became difficult, we have migrated the project template generator directly into the [Isaac Lab](https://github.com/isaac-sim/IsaacLab) repository.

👉 Please follow the steps in the Isaac Lab docs to set up your own project: [How to create your own Isaac Lab project](https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/index.html)

If you would like to contribute to the extension template, please follow the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html) in the Isaac Lab repository directly.
 
---

# Template for Isaac Lab Projects

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

This repository serves as a template for building projects or extensions based on Isaac Lab. It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Parallel welding scene

Generate workpiece/path jobs:

```bash
python data_generation/src/main.py --count 4 --jobs-dir data_generation/data/generated_jobs --spacing 2.0
```

Generate a 5x5 set of 25 workstations:

```bash
python data_generation/src/main.py \
  --count 25 \
  --jobs-dir data_generation/data/generated_jobs \
  --layout grid \
  --grid-cols 5 \
  --spacing 2.0
```

Import the generated STL workpieces and spawn one UR5e welding arm per job:

```bash
python scripts/sim_parallel_welding.py \
  --manifest data_generation/data/generated_jobs/manifest.json
```

Move only the workpiece relative to the robot/env origin:

```bash
python scripts/sim_parallel_welding.py \
  --manifest data_generation/data/generated_jobs/manifest.json \
  --workpiece-offset 0.5 0.0 0.0
```

`origin` in the manifest moves the whole robot-workpiece cell. `workpiece_offset` moves only the STL workpiece inside that cell, in meters.

Record a top-down view of the 5x5 scene:

```bash
python scripts/sim_parallel_welding.py \
  --manifest data_generation/data/generated_jobs/manifest.json \
  --headless \
  --record \
  --num-steps 10 \
  --keep-frames \
  --camera-eye 4.0 4.0 14.0 \
  --camera-target 4.0 4.0 0.0 \
  --frames-dir outputs/top_25_frames \
  --output outputs/top_25.mp4 \
  --overwrite
```

For server/headless runs:

```bash
python scripts/sim_parallel_welding.py \
  --manifest data_generation/data/generated_jobs/manifest.json \
  --headless
```

Record a headless MP4 for inspection:

```bash
python scripts/sim_parallel_welding.py \
  --manifest data_generation/data/generated_jobs/manifest.json \
  --headless \
  --record \
  --num-steps 180 \
  --fps 30 \
  --output outputs/parallel_welding_scene.mp4 \
  --overwrite
```

Keep the intermediate PNG frames:

```bash
python scripts/sim_parallel_welding.py \
  --manifest data_generation/data/generated_jobs/manifest.json \
  --headless \
  --record \
  --num-steps 1 \
  --keep-frames \
  --output outputs/parallel_welding_snapshot.mp4 \
  --overwrite
```

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
# Option 1: HTTPS
git clone https://github.com/isaac-sim/IsaacLabExtensionTemplate.git

# Option 2: SSH
git clone git@github.com:isaac-sim/IsaacLabExtensionTemplate.git
```

- Throughout the repository, the name `weldRobot` only serves as an example and we provide a script to rename all the references to it automatically:

```bash
# Enter the repository
cd IsaacLabExtensionTemplate
# Rename all occurrences of weldRobot (in files/directories) to your_fancy_extension_name
python scripts/rename_template.py your_fancy_extension_name
```

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/weldRobot
```

- Verify that the extension is correctly installed by running the following command:

```bash
python scripts/rsl_rl/train.py --task=Template-Isaac-Velocity-Rough-Anymal-D-v0
```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu. When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory. The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse. This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/weldRobot/weldRobot/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of your repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon** (☰), then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to `IsaacLabExtensionTemplate/source`
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon** (☰), then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Docker setup

### Building Isaac Lab Base Image

Currently, we don't have the Docker for Isaac Lab publicly available. Hence, you'd need to build the docker image
for Isaac Lab locally by following the steps [here](https://isaac-sim.github.io/IsaacLab/main/source/deployment/index.html).

Once you have built the base Isaac Lab image, you can check it exists by doing:

```bash
docker images

# Output should look something like:
#
# REPOSITORY                       TAG       IMAGE ID       CREATED          SIZE
# isaac-lab-base                   latest    28be62af627e   32 minutes ago   18.9GB
```

### Building Isaac Lab Template Image

Following above, you can build the docker container for this project. It is called `isaac-lab-template`. However,
you can modify this name inside the [`docker/docker-compose.yaml`](docker/docker-compose.yaml).

```bash
cd docker
docker compose --env-file .env.base --file docker-compose.yaml build isaac-lab-template
```

You can verify the image is built successfully using the same command as earlier:

```bash
docker images

# Output should look something like:
#
# REPOSITORY                       TAG       IMAGE ID       CREATED             SIZE
# isaac-lab-template               latest    00b00b647e1b   2 minutes ago       18.9GB
# isaac-lab-base                   latest    892938acb55c   About an hour ago   18.9GB
```

### Running the container

After building, the usual next step is to start the containers associated with your services. You can do this with:

```bash
docker compose --env-file .env.base --file docker-compose.yaml up
```

This will start the services defined in your `docker-compose.yaml` file, including isaac-lab-template.

If you want to run it in detached mode (in the background), use:

```bash
docker compose --env-file .env.base --file docker-compose.yaml up -d
```

### Interacting with a running container

If you want to run commands inside the running container, you can use the `exec` command:

```bash
docker exec --interactive --tty -e DISPLAY=${DISPLAY} isaac-lab-template /bin/bash
```

### Shutting down the container

When you are done or want to stop the running containers, you can bring down the services:

```bash
docker compose --env-file .env.base --file docker-compose.yaml down
```

This stops and removes the containers, but keeps the images.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/weldRobot"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
