# Isaac Sim job output

`src/main.py` now packages generated welding data for parallel Isaac Sim runs.

Run:

```bash
python src/main.py --count 8 --jobs-dir data/generated_jobs --spacing 2.0
```

Simple-structure dataset:

```bash
python src/main.py --simple-structures --simple-samples-per-type 30
```

This writes simple-structure jobs to `data/generated_jobs/simple_jobs/` by default and keeps the original random-workpiece flow unchanged unless `--simple-structures` is passed.

The pipeline still writes the original intermediate outputs:

- `data/model/*.step` and `data/model/*.stl`
- `data/extract/<model_id>/...`
- `data/path/*.json`
- `data/vector/*_weld_vectors.json`

It also writes a simulation-ready package:

```text
data/generated_jobs/
  manifest.json
  job_000/
    workpiece.step
    workpiece.stl
    path.json
    weld_vectors.json
    raw_weld_topology.json
  job_001/
    ...
```

`manifest.json` paths are relative to the manifest directory. Each job contains one workpiece and one path:

```json
{
  "id": "job_000",
  "workpiece_asset": "job_000/workpiece.stl",
  "path_json": "job_000/path.json",
  "origin": [0.0, 0.0, 0.0],
  "frame": "workpiece",
  "units": "mm"
}
```

`path.json` is the controller-facing path format:

```json
{
  "schema": "weld_robot.path.v1",
  "frame": "workpiece",
  "units": "mm",
  "orientation": "normal_vector",
  "segments": [],
  "waypoints": []
}
```

The original full weld topology is preserved as `raw_weld_topology.json` for debugging and algorithm development.
