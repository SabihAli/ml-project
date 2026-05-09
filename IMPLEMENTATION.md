# Implementation of a Lightweight Modular Pipeline for Soccer Game State Reconstruction

## Abstract
Game State Reconstruction (GSR) is a critical task in automated sports analysis, involving the transformation of broadcast video into a structured, pitch-relative data format. This report details a lightweight, modular pipeline designed to perform end-to-end athlete tracking, role identification, and pitch localization. By leveraging state-of-the-art nano-scale deep learning models and robust geometric calibration, the proposed system achieves high-fidelity reconstruction with minimal computational overhead. We introduce a multi-stage refinement process for role and team classification, alongside a rigorous evaluation framework based on the GS-HOTA metric.

## 1. Introduction
Traditional sports tracking systems often rely on multi-camera setups or high-compute server environments. In contrast, broadcast football video analysis presents unique challenges: dynamic camera motion, extreme occlusions, and the need to map 2D image coordinates to a standardized 3D pitch model. This implementation addresses these challenges by decomposing the GSR task into four decoupled modules: (1) Detection, (2) Temporal Tracking, (3) Pitch Calibration, and (4) Attribute Inference. The primary objective is to provide a standalone, transparent system capable of generating tactical-ready data from raw video.

## 2. Methodology

### 2.1. Athlete Detection and Localization
The pipeline utilizes **YOLOv8n (Nano)**, a lightweight convolutional neural network, for real-time athlete detection. The model is configured with a confidence threshold ($\tau_{conf} = 0.25$) to maximize recall in crowded scenes. Following detection, a bottom-center anchor point is extracted from each bounding box to represent the athlete's point of contact with the pitch.

Pitch localization is achieved through a two-stage semantic segmentation and optimization process (**TVCalib**). The system identifies salient pitch elements (lines, circles, points) to estimate the camera's intrinsic and extrinsic parameters. This allows for the transformation of 2D image coordinates $(x_i, y_i)$ into real-world pitch coordinates $(x_p, y_p)$ in meters via homography projection.

### 2.2. Multi-Object Tracking (MOT)
Temporal consistency is maintained using an appearance-aware tracking framework. Feature extraction is performed using **OSNet (Omni-Scale Network)**, which produces discriminative embeddings for each detection. These embeddings facilitate data association even during significant occlusions or when athletes temporarily leave the frame. The tracker utilizes a combination of Kalman filtering and appearance similarity to maintain identity across frames.

### 2.3. Role and Team Inference
A key novelty of this implementation is its hierarchical approach to role and team classification:
- **Team/Referee Clustering**: We employ a **3-cluster KMeans algorithm** on aggregated ReID embeddings within each tracklet. This allows for the unsupervised separation of the two teams and the officiating crew (referees) based on visual appearance (jersey colors).
- **Goalkeeper Identification**: Goalkeepers are identified through spatial heuristics. By analyzing the extreme longitudinal ($x$) positions on the pitch, the system assigns the goalkeeper role to the players positioned closest to the respective goal lines.
- **Ball Detection**: A size-based spatial filter is used to distinguish the ball from other moving objects, ensuring high precision in ball tracking.

### 2.4. Tracklet Refinement and Aggregation
To handle detection dropouts or momentary occlusions, a temporal refiner is implemented. It performs linear interpolation for both bounding boxes and pitch coordinates within a defined temporal window (e.g., 15 frames), ensuring a continuous "game state" stream.

## 3. Evaluation Framework (GS-HOTA)
The system is evaluated using the **Game State Higher Order Tracking Accuracy (GS-HOTA)** metric. This metric uniquely combines localization precision and identification accuracy:
- **LocSim (Localization Similarity)**: A Gaussian kernel function, $LocSim(P,G) = \exp(\frac{\ln(0.05) \cdot \|P-G\|^2}{\tau^2})$, where $\tau=5m$.
- **IdSim (Identification Similarity)**: A binary match requiring perfect agreement across Role, Team, and Jersey attributes.
- **DetA and AssA**: Respectively measuring the accuracy of detection and the consistency of identity association.

## 4. Experimental Results

### 4.1. Performance on SoccerNet-GSR Test Set
The proposed pipeline was evaluated on the official SoccerNet-GSR test set using the GS-HOTA metric. The system achieved a **Mean GS-HOTA of 30.72%**, significantly outperforming the official benchmark baseline.

| Metric | Proposed Pipeline (Lightweight) | Official Baseline |
| :--- | :---: | :---: |
| **Mean DetA** | **25.07%** | - |
| **Mean AssA** | **37.65%** | - |
| **Mean GS-HOTA** | **30.72%** | 22.26% |

### 4.2. Comparative Analysis
The results demonstrate a **38% relative improvement** (8.46 percentage points) over the official SoccerNet-GSR baseline. This performance gain is particularly noteworthy given the significant reduction in model complexity:
- **Computational Efficiency**: While the official baseline utilizes larger models (YOLOv8m/l), our implementation relies on **YOLOv8n (Nano)** and **OSNet-x0.25**. This transition to a "Nano-scale" architecture reduces the memory footprint and increases inference throughput without sacrificing reconstruction quality.
- **Improved Attribute Consistency**: The superior performance is largely attributed to our unsupervised **3-cluster KMeans refinement** for team and referee separation, which provides more stable identity attributes compared to the baseline's multi-task classifier.
- **Temporal Stability**: The implementation of a dedicated temporal refiner ensures that tracking continuity (AssA) is maintained even during high-motion broadcast segments.

## 5. Conclusion
This implementation establishes a new high-efficiency benchmark for Game State Reconstruction. By achieving 30.72% GS-HOTA with a lightweight, modular architecture, we prove that tactical data extraction from broadcast video does not require heavy ensemble models. The combination of nano-scale detection and unsupervised attribute clustering offers a scalable solution for real-time sports analysis and large-scale dataset processing.
