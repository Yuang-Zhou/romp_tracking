# Simple_ROMP

Simple implementation of ROMP [ICCV21], BEV [CVPR22], and TRACE [CVPR23].

- **ROMP**: Lightweight head for estimating SMPL 3D pose/shape and rough 2D position/scale.
- **BEV**: Adds depth reasoning and SMPL+A (all ages) support.
- **TRACE**: Tracks selected subjects over time in global coordinates. ([TRACE instructions](trace2/README.md))

## Installation

1. Install dependencies and package:
```bash
pip install --upgrade setuptools numpy cython lapx
pip install simple_romp==1.1.4
````

or from source:

```bash
python setup.py install
```

2. (Mac) Upgrade PyTorch for BEV support:

```bash
pip install --upgrade torch torchvision
```

## SMPL Model Preparation

1. Download:

   * [Meta data](https://github.com/Arthur151/ROMP/releases/download/V2.0/smpl_model_data.zip)
   * [SMPL model (SMPL\_NEUTRAL.pkl)](https://smpl.is.tue.mpg.de/)
   * (Optional for BEV) [SMIL model](https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html)

2. Folder structure:

```
smpl_model_data/
 ├── SMPL_NEUTRAL.pkl
 ├── J_regressor_extra.npy
 ├── J_regressor_h36m.npy
 ├── smpl_kid_template.npy
 └── smil/
     └── smil_web.pkl
```

3. Convert models:

```bash
romp.prepare_smpl -source_dir=/path/to/smpl_model_data
bev.prepare_smil -source_dir=/path/to/smpl_model_data  # optional
```

Results will be saved to `~/.romp/`:

```
.romp/
 ├── SMPL_NEUTRAL.pth
 ├── SMPLA_NEUTRAL.pth
 └── smil_packed_info.pth
```

## Usage

### Webcam demo

```bash
romp --mode=webcam --show
bev --mode=webcam --show
```

### Process single image

```bash
romp --mode=image --calc_smpl --render_mesh -i=/path/to/image.jpg -o=/path/to/results.jpg
bev -i /path/to/image.jpg -o /path/to/results.jpg
```

### Process folder of images

```bash
romp --mode=video --calc_smpl --render_mesh -i=/path/to/image/folder/ -o=/path/to/output/folder/
bev -m video -i /path/to/image/folder/ -o /path/to/output/folder/
```

### Process video file

```bash
romp --mode=video --calc_smpl --render_mesh -i=/path/to/video.mp4 -o=/path/to/output/results.mp4 --save_video
bev -m video -i /path/to/video.mp4 -o /path/to/output/results.mp4 --save_video
```

### Optional flags

* `--show`: Display results during processing
* `--show_items=mesh,mesh_bird_view`: Customize visualization
* `-t -sc=3.`: Smoothing (ROMP only)
* `--onnx`: Faster CPU inference (ROMP only)
* `--show_largest`: Focus on largest subject (ROMP only)

## Python API

```python
import romp, cv2
settings = romp.main.default_settings
romp_model = romp.ROMP(settings)
outputs = romp_model(cv2.imread('image.jpg'))

import bev
settings = bev.main.default_settings
bev_model = bev.BEV(settings)
outputs = bev_model(cv2.imread('image.jpg'))
```

## Export

See [`export.md`](doc/export.md) for exporting to `.fbx`, `.glb`, `.bvh`.

## Convert Checkpoints

```bash
cd /path/to/ROMP/simple_romp/
python tools/convert_checkpoints.py ROMP.pkl ROMP.pth
```

## Load Results from `.npz`

```python
import numpy as np
results = np.load('results.npz', allow_pickle=True)['results'][()]
```

## Output Joints

71 joints = 24 SMPL + 30 extra + 17 H36m joints.
See source code for joint mappings (`SMPL_24`, `SMPL_EXTRA_30`).

## License

MIT License.# romp_tracking
