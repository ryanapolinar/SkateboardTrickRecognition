# Skateboard Trick Recognition System Architecture

## 1. Project Scope

This document outlines the technical design for a lightweight, real-time computer vision pipeline engineered to identify skateboarding tricks from video clips. By relying on keypoint pose tracking and board trajectory analysis, the architecture deliberately avoids heavy, scene-memorizing 3D-CNNs. This approach ensures low latency, high interpretability, and robust performance across varying environmental backgrounds.

## 2. Tech Stack

The core components of the pipeline prioritize efficiency and low-latency frame inference:

| Component           | Library / Framework                                   | Technical Scope & Purpose                                                                         |
| :------------------ | :---------------------------------------------------- | :------------------------------------------------------------------------------------------------ |
| Pose Estimation     | ultralytics (YOLOv8-Pose / YOLOv11-Pose) or mediapipe | Tracks 2D spatial keypoints for feet, knees, and hips across consecutive frames.                  |
| Board Detection     | YOLOv8 / YOLOv11 (Custom fine-tuned)                  | Detects and tracks board bounding boxes and orientation features (Nose vs. Tail).                 |
| Object Tracking     | supervision / ByteTrack                               | Maintains board and rider temporal identity through high-velocity movement and severe occlusions. |
| Sequence Classifier | PyTorch (nn.LSTM, TCN) or xgboost                     | Classifies time-series feature vectors extracted across rolling frame windows.                    |
| Video Utility       | opencv-python                                         | Manages frame capture, stream decoding, ROI cropping, and visual output rendering.                |

## 3. Data Representation (Frame Vector Strategy)

Instead of feeding raw pixel sequences into heavy spatiotemporal models, the pipeline converts raw frames into a compact sequence of spatial feature vectors. Processing is bounded within a rolling 30 to 45 frame window (~1.0 to 1.5 seconds at 30 FPS).

### Feature Vector Composition

Each frame is abstracted into a 7-dimensional feature vector:

1. hip_y_position: Vertical coordinate of the rider's hip center (tracks pop elevation).
2. left_ankle_board_dist: Euclidean distance between the left ankle keypoint and board center.
3. right_ankle_board_dist: Euclidean distance between the right ankle keypoint and board center.
4. board_center_x: Normalized horizontal centroid coordinate of the skateboard.
5. board_center_y: Normalized vertical centroid coordinate of the skateboard.
6. board_aspect_ratio: Ratio of width to height (W / H) of the board bounding box (indicates flip/roll state).
7. board_angle: Orientation angle of the main board axis relative to horizontal.

### Tensor Specifications

- Input Tensor Shape: (Batch_Size, Time_Steps=30, Num_Features=7)

## 4. Hierarchical Classification Strategy

To minimize computational overhead, trick identification operates as a tiered decision tree, triggering detailed feature evaluation only when an initial pop event is verified.

```text
[ Video Frame Stream ]
          │
          ▼
[ Stage 1: Pop Event Detector ]
(Airtime Trigger via Hip/Ankle Y)
          │
          ├────────────────────────┐
       (No Pop)                 (Pop Detected)
          │                        │
          ▼                        ▼
    [ Ignore Frame ]     [ Stage 2: Board Motion Check ]
                         (Aspect Ratio & Angle Variance)
                                   │
                  ┌────────────────┴────────────────┐
           (Ratio Static)                   (Rotation Detected)
                  │                                 │
                  ▼                                 ▼
           [ Stage 3a: Ollie ]             [ Stage 3b: Spin / Flip ]
                                                    │
                                   ┌────────────────┴────────────────┐
                              (Roll Axis)                       (Yaw Axis)
                                   │                                 │
                                   ▼                                 ▼
                          [ Kickflip / Heelflip ]            [ Pop Shuvit ]
```
