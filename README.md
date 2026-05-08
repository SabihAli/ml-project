# SoccerNet Game State Recognition (Modular GSR)

A standalone, lightweight reimplementation of the SoccerNet Game State Reconstruction (GSR) baseline. This pipeline performs end-to-end athlete tracking, identification, and pitch localization from broadcast football videos.

## 🚀 Key Features

- **Lightweight & Fast**: Uses YOLOv8n (Nano) and OSNet for efficient processing on standard hardware.
- **Official GS-HOTA Evaluation**: Built-in benchmarking module strictly following the SoccerNet-GSR standards.
- **Pitch Localization**: Maps players from 2D image coordinates to real-world pitch coordinates (meters).
- **Modular Design**: Separated modules for Detection, Tracking, Team Classification, Jersey Recognition, and Refinement.
- **No Heavy Dependencies**: Removed reliance on the complex TrackLab/Hydra framework for better transparency and speed.

---

## 🛠️ Installation & Setup

### 1. Create a Virtual Environment
```bash
# Create the environment
python -m venv .gsr_venv

# Activate the environment (Windows)
.gsr_venv\Scripts\activate

# Activate the environment (Linux/macOS)
source .gsr_venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements_pipeline.txt
```

### 3. Setup Calibration Plugin
The `nbjw_calib` package is required for pitch mapping.
```bash
cd gsr_pipeline/nbjw_calib
pip install -e .
cd ../..
```

---

## 🏃 Usage

### 1. Run Inference
Process a SoccerNet sequence and generate predictions:
```bash
python -m gsr_pipeline.run --sequence_dir data/SoccerNetGS/gamestate-2024/valid/SNGS-021 --output_dir outputs/SNGS-021 --max_frames 50
```
- `--sequence_dir`: Path to the sequence folder (containing `img1/`).
- `--output_dir`: Where to save `predictions.json` and `annotated_video.mp4`.
- `--max_frames`: (Optional) Limit processing to the first N frames.

### 2. Evaluate Performance (GS-HOTA)
Calculate official metrics against ground truth:
```bash
python -m gsr_pipeline.evaluate --gt_path path/to/Labels-GameState.json --pred_path outputs/SNGS-021/predictions.json
```
This generates a detailed report including **DetA**, **AssA**, and the final **GS-HOTA** score.

---

## 📊 Evaluation Metrics

The pipeline uses the **GS-HOTA** metric, which combines:
- **DetA (Detection Accuracy)**: Measures how well players are localized and correctly identified (Role, Team, Jersey).
- **AssA (Association Accuracy)**: Measures tracking consistency across frames.
- **LocSim**: Gaussian similarity (tau=5m) for pitch localization.
- **IdSim**: Binary matching for Role, Team, and Jersey Number.

---

## ⚙️ Configuration

Module parameters (thresholds, model paths, clustering) are managed in:
`gsr_pipeline/configs/default.yaml`

Key settings:
- `detector.conf_threshold`: Default is `0.25`.
- `team.n_clusters`: Set to `3` (separates Team A, Team B, and Referees).
- `reid.model_name`: Default is `osnet_x0_25`.

---

## 📂 Project Structure

```text
├── gsr_pipeline/          # Core pipeline source code
│   ├── configs/           # YAML configuration files
│   ├── detect.py          # YOLOv8 Athlete Detection
│   ├── track.py           # Multi-Object Tracking
│   ├── team.py            # KMeans Team/Referee Classification
│   ├── jersey.py          # EasyOCR Jersey Recognition
│   ├── evaluate.py        # GS-HOTA Metric Implementation
│   └── run.py             # Main execution script
├── data/                  # Dataset directory (SoccerNetGS)
├── outputs/               # Prediction and visualization results
└── requirements_pipeline.txt
```

## 📝 License
This project follows the licensing of the original SoccerNet-GSR baseline and associated models (YOLOv8, TorchReID).
