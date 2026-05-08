# SoccerNet GSR Pipeline (Clean Reimplementation)

A standalone, lightweight reimplementation of the SoccerNet Game State Reconstruction baseline.

## Key Changes
- **YOLOv8n**: Nano detector (~6MB) instead of the original heavy YOLOv8m/11m.
- **OSNet ReID**: Lightweight feature extractor via `torchreid` instead of the heavy `prtreid` stack.
- **EasyOCR**: Simplified jersey number detection.
- **ByteTrack**: Using the built-in Ultralytics tracker.
- **No tracklab/Hydra**: Plain Python orchestration for better transparency and speed.

## Installation & Setup

Follow these steps to set up a clean environment for the GSR pipeline.

### 1. Create a Virtual Environment
It's recommended to use a virtual environment to avoid dependency conflicts.
```bash
# Create the environment
python -m venv .gsr_venv

# Activate the environment
# On Windows:
.gsr_venv\Scripts\activate
# On Linux/macOS:
source .gsr_venv/bin/activate
```

### 2. Install Core Dependencies
Ensure the environment is activated, then install the necessary packages.
```bash
pip install -r requirements_pipeline.txt
```

### 3. Install Calibration Plugin
The `nbjw_calib` package is required for pitch calibration.
```bash
cd sn-gamestate/plugins/calibration
pip install -e .
cd ../../..
```

## Usage

Once the environment is set up and activated, you can run the pipeline on any SoccerNet sequence.

### Basic Run
Run the pipeline on the first 10 frames of a validation sequence:
```bash
python -m gsr_pipeline.run --sequence_dir data/SoccerNetGS/gamestate-2024/valid/SNGS-021 --output_dir outputs/SNGS-021 --max_frames 10
```

### Arguments
- `--sequence_dir`: Path to the SoccerNet sequence directory.
- `--output_dir`: Where to save the `predictions.json`.
- `--max_frames`: (Optional) Limit the number of frames to process.
- `--device`: (Optional) Override the device in the config (e.g., `cuda` or `cpu`).

## Configuration
All module parameters (thresholds, model paths, devices) are managed in a single YAML file:
[gsr_pipeline/configs/default.yaml](file:///c:/Users/sabih/OneDrive/Desktop/game_state_recognition/gsr_pipeline/configs/default.yaml)

## Environment Management
- **Deactivate**: Run `deactivate` to exit the virtual environment.
- **Re-activate**: Run the activation command for your OS (see step 1) whenever you start a new terminal session.
- **Updates**: If `requirements_pipeline.txt` changes, run `pip install -r requirements_pipeline.txt` again.
