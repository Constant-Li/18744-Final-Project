# Detection of Active Turn Signals, Brake Lights, and Flashers

End-to-end 18-744 Autonomous Driving final project for detecting, tracking, and recognizing vehicle light signals in traffic videos. The project combines a YOLO vehicle detector, BoT-SORT tracking, crop caching, frame-wise recognition, causal post-processing, video-wise temporal modeling, and head/tail orientation recognition.

## Demo

[Watch the demo video](https://docs.google.com/file/d/1389s3JKsW4pIslVcairsRDP3mDmQcRnt/preview)

## Project Summary

Vehicle light signals provide explicit early intent cues that motion-only perception often sees too late. Brake lights can appear before measurable deceleration, turn indicators usually precede lateral movement, and hazard flashers communicate abnormal stopping or caution states. This project treats light-state recognition as a dedicated perception problem and outputs vehicle-level temporal predictions: which tracked vehicle signaled what, and when.

The final system supports real-world RGB driving video, multi-vehicle localization, stable track IDs, per-vehicle ROI extraction, turn/brake classification, front/rear orientation recognition, cached playback, and JSON outputs for debugging or downstream analysis.

## What This Project Does

The main application is `final_demo_cached.py`. It provides a PyQt interface that:

1. Loads a traffic video.
2. Detects vehicles with a YOLO model.
3. Tracks vehicles across frames with BoT-SORT.
4. Saves per-vehicle crops in frame-wise and video-wise folders.
5. Runs recognition models for:
   - turn signal: `off`, `left`, `right`, `both`, `unknown`
   - brake light: `brake_off`, `brake_on`
   - vehicle view: `head`, `tail`
6. Applies causal real-time post-processing to smooth frame-wise predictions.
7. Displays the tracked video, vehicle crops, and recognition outputs in the GUI.

## Architecture

The runtime pipeline has five main stages:

1. Vehicle detection and tracking: an Ultralytics YOLO detector localizes vehicles, and BoT-SORT links detections into stable vehicle tracks.
2. Crop generation and caching: per-track crops are saved frame-wise and sequence-wise so recognition can be rerun without repeating detection/tracking.
3. Frame-wise recognition: a ResNet34 multi-head classifier predicts brake and turn states from each 224x224 vehicle crop.
4. Temporal reasoning: causal post-processing smooths frame-wise outputs using EMA and hold logic; the video-wise branch uses a ResNet34 + BiLSTM model for turn-signal blinking patterns.
5. Head/tail orientation: a separate ResNet34 classifier predicts whether each crop is front-view or rear-view to make signal interpretation more reliable.

## Repository Layout

```text
.
|-- final_demo_cached.py              # Current integrated demo app
|-- midterm_demo.py                   # Earlier GUI demo version
|-- midterm_demo_pp.py                # Midterm demo with post-processing
|-- yolo_training.py                  # YOLO fine-tuning script
|-- botsort_reid_vehicle.yaml         # BoT-SORT tracker config used by the demo
|-- Train_and_Test/                   # Main PyTorch training/evaluation pipeline
|-- headtail/                         # Head/tail classifier training pipeline
|-- labeling_tools/                   # PyQt tools for vehicle/view/light labeling
|-- data_processing_mics/             # Dataset conversion and post-processing utilities
|-- training model/                   # Experimental training scripts and patches
|-- BoT-SORT/                         # Vendored BoT-SORT, YOLOX, YOLOv7, FastReID code
|-- archieved_yolo_weights/           # Archived detector weights and training artifacts
`-- runs/                             # YOLO training outputs and evaluation plots
```

## Environment

Python 3.8+ is recommended. The project uses PyTorch, TorchVision, OpenCV, Ultralytics YOLO, PyQt5, NumPy, and the BoT-SORT/FastReID dependency stack.

Create an environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision
pip install ultralytics opencv-python PyQt5 numpy pyyaml tqdm matplotlib scikit-learn pillow
pip install -r BoT-SORT/requirements.txt
```

Install the correct PyTorch build for your CUDA version if you plan to run on GPU.

## Important Paths To Configure

Several scripts currently contain absolute local paths from development. Update them before running on a new machine:

- `final_demo_cached.py`
  - `DET_MODEL_PATH` should point to the YOLO detector weight, usually a `best.pt`.
  - `TRACKER_CFG` should point to `botsort_reid_vehicle.yaml`.
- `yolo_training.py`
  - `data_yaml` should point to the YOLO dataset YAML.
  - `YOLO(...)` should point to the starting detector weight.
- `Train_and_Test/datasets/*.py` and `headtail/datasets/*.py`
  - update dataset prefixes such as `/Users/yyf/Mine_Space/18744/project/dataset`.

The trained recognition weights are not hard-coded in the final GUI. The app asks you to select `.pt`, `.pth`, or `.ckpt` files when running each recognition model.

## Key Runtime Parameters

`final_demo_cached.py` exposes the most important runtime knobs near the top of the file:

```text
DET_CONF = 0.50              # detector confidence threshold
TRUST_CONF = 0.60            # confidence threshold for trusted crop saving
IOU_THRES = 0.7              # detector/tracker IoU threshold
RECOG_INPUT_SIZE = 224       # recognition crop size
FRAME_WISE_STRIDE = 1        # process every frame for smoother demo playback
VIDEO_WISE_STRIDE = 16       # group size for video-wise crop folders
```

Increasing `FRAME_WISE_STRIDE` can speed up inference, but produces less dense recognition output.

## Running The Final Demo

```powershell
python final_demo_cached.py
```

In the GUI:

1. Click `Select Video and Run Detect+Track`.
2. Choose an input video.
3. Choose an output directory.
4. Wait for tracking to finish.
5. Run frame-wise, video-wise, or head/tail recognition and select the corresponding model weights when prompted.

The app writes outputs under:

```text
<chosen_output_dir>/output/<video_name>/
|-- frame-wise/                    # per-frame vehicle crops grouped by track id
|-- video-wise/                    # 16-frame crop sequences grouped by track id
|-- visualization/                 # tracked visualization video
|-- tracking_meta.json
|-- frame_wise_recognition.json
|-- video_wise_recognition.json
|-- head_tail_recognition.json
`-- postprocess_recognition.json
```

If the same video has already been tracked, the app can reuse the cached tracking bundle instead of recomputing it.

## Training Recognition Models

The main recognition training code is in `Train_and_Test/`.

Frame-wise brake/turn classifier:

```powershell
cd Train_and_Test
python main.py --config configs/TLD.yaml
```

Video-wise turn classifier:

```powershell
cd Train_and_Test
python main.py --config configs/TLD_video.yaml
```

Head/tail classifier:

```powershell
cd headtail
python main.py --config configs/TLD.yaml
```

Model checkpoints are saved to the configured `work_dir`, typically as `cur_test_model.pt`.

## Evaluation Highlights

Final report metrics on the test set:

```text
Method             Causality   Brake    Left Turn   Right Turn   Hazard Turn   Off
Frame-wise         Causal      93.15%   74.51%      72.79%       68.89%        96.80%
Post-processing    Causal      94.08%   79.22%      77.93%       73.36%        96.61%
Video-wise         Non-causal  N/A      79.96%      78.14%       71.28%        96.57%
```

Additional reported results:

```text
Head/tail classifier accuracy: 96.97%
YOLOv26s detector after 60 epochs: precision 95.78%, recall 83.68%, mAP@50 91.52%, mAP@50-95 84.19%
```

The main takeaway is that temporal reasoning matters most for active turn signals. Causal post-processing improves frame-wise brake, left, right, and hazard recognition while remaining suitable for online use; the video-wise model is stronger for turn patterns but is non-causal and currently turn-only.

## Training The YOLO Detector

Update the paths in `yolo_training.py`, then run:

```powershell
python yolo_training.py
```

The script fine-tunes a YOLO model with 640px inputs and writes results under `runs_detect/ft_bestpt_640`. The best detector weight is saved as:

```text
runs_detect/ft_bestpt_640/weights/best.pt
```

## Data Processing Utilities

Remap vehicle orientation labels into two YOLO classes, `vehicle_back` and `vehicle_front`:

```powershell
python data_processing_mics/preprocess_vehicle_orientation_labels.py <input_label_dir> --output-dir <output_label_dir>
```

Track annotated TLD frames and assign stable vehicle IDs:

```powershell
python data_processing_mics/tld_track.py ^
  --input_json <input.json> ^
  --output_json <tracked_output.json> ^
  --image_root <image_root> ^
  --botsort_repo BoT-SORT ^
  --tracker_cfg botsort_reid_vehicle.yaml
```

Simplify demo post-processing outputs:

```powershell
python data_processing_mics/post_processing_json.py <output_root> --combined-name all_simple_postprocess.json
```

## Labeling Tools

The `labeling_tools/` folder contains PyQt annotation helpers:

- `vehicle_view_labeling.py` detects/crops vehicles from image folders and labels vehicle view/orientation.
- `vehicle_light_labeling.py` tracks vehicles in a video and labels turn/brake lights by frame or interval.

Run them directly with Python:

```powershell
python labeling_tools/vehicle_view_labeling.py
python labeling_tools/vehicle_light_labeling.py
```

## Models And Labels

Core recognition models:

- `Train_and_Test/light_network.py`: ResNet34 frame-wise classifier for turn and brake labels.
- `Train_and_Test/TLD_video_network.py`: ResNet34 + BiLSTM video-wise turn classifier.
- `headtail/light_network.py`: ResNet34 head/tail classifier.

Primary label sets:

```text
Turn:  off, left, right, both, unknown
Brake: brake_off, brake_on
View:  head, tail
```

## Known Limitations

- Cross-domain generalization remains limited when videos come from new cameras, viewpoints, or lighting distributions.
- The recognition training data is biased toward taillight views, so front-view signal recognition needs stronger dedicated data.
- Full detection, tracking, and video-wise recognition are computationally expensive for strict real-time deployment.
- Video-wise labels are built from fixed temporal windows, which can blur signal onset and offset boundaries.
- Tracking failures such as ID switches or track fragmentation can corrupt the temporal signal history for a vehicle.

## Notes

- `BoT-SORT/` contains third-party tracking, YOLOX, YOLOv7, and FastReID code. Its own README and licenses are preserved inside that folder.
- `archieved_yolo_weights/` and `runs/` contain generated model weights, plots, and training artifacts.
- `__pycache__/`, `.pyc`, image outputs, videos, and `.pt/.pth` files are generated or binary artifacts rather than source code.
- The current project is research-oriented, so several scripts are exploratory and contain absolute paths. Treat `final_demo_cached.py`, `Train_and_Test/`, `headtail/`, `labeling_tools/`, and `data_processing_mics/` as the main project surface.
