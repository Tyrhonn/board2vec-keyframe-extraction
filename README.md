# BOARD2VEC Keyframe Extraction

Shot boundary detection using **TransNetV2** and representative keyframe extraction using **OpenCV** for the BOARD2VEC project.

---

## 🛠️ Project Structure

```text
board2vec-keyframe-extraction/
├── extract_keyframes.py       # Keyframe extraction script
├── requirements.txt           # Python dependencies
├── README.md                  # Setup and usage guide
└── .gitignore                 # Excluded files and datasets
```

**Note:** TransNetV2 is maintained separately and is not included in this repository.

---

## 🚀 Environment Setup & Installation

### 1. Prerequisites

- **Python:** 3.12 (tested locally)
- **Git:** For cloning repositories
- **FFmpeg:** For TransNetV2 video inference
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`

### 2. Clone Repositories

Clone TransNetV2 and the keyframe extraction repository into the same directory:

```bash
git clone https://github.com/soCzech/TransNetV2.git
git clone <YOUR_REPOSITORY_URL>
```

Expected structure:

```text
workspace/
├── TransNetV2/
└── board2vec-keyframe-extraction/
```

### 3. Set Up TransNetV2

```bash
cd TransNetV2

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install tensorflow==2.21.0 ffmpeg-python pillow
```

Download the pretrained weights according to the [official TransNetV2 inference instructions](https://github.com/soCzech/TransNetV2/tree/master/inference).

> TensorFlow 2.21.0 was used in the local macOS experiment. Compatibility on other systems has not been verified.

### 4. Set Up Keyframe Extraction

```bash
deactivate
cd ../board2vec-keyframe-extraction

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
```

The extraction dependencies are NumPy and OpenCV. TransNetV2 requires a separate setup.

---

## 🎬 Extraction Pipeline

```text
Input Video (.mp4)
       |
       v
TransNetV2 Shot Detection
       |
       v
Shot Boundaries (.scenes.txt)
       |
       v
Temporal Window Sampling
       |
       v
MAD-based Frame Comparison
       |
       v
Medoid Selection
       |
       v
Redundancy Filtering
       |
       v
Keyframes (.png) + CSV Metadata
```

---

## 📂 Dataset Structure

Input videos and scene files must be placed inside `dataset/videos/`:

```text
dataset/
├── videos/
│   ├── vid_0001.mp4
│   ├── vid_0001.mp4.scenes.txt
│   └── ...
└── keyframes/
```

Dataset files are excluded from Git.

---

## ▶️ Usage

### 1. Detect Shot Boundaries

From the `TransNetV2/` directory:

```bash
source .venv/bin/activate

python inference/transnetv2.py \
  ../board2vec-keyframe-extraction/dataset/videos/vid_0001.mp4
```

This generates a `.scenes.txt` file containing detected shot boundaries, alongside prediction scores.

### 2. Extract Keyframes

Switch to the extraction repository:

```bash
deactivate
cd ../board2vec-keyframe-extraction
source .venv/bin/activate
```

Review the video selection and CSV writing configuration in `extract_keyframes.py`, then run:

```bash
python extract_keyframes.py
```

### 3. Output

```text
dataset/
├── keyframes/
│   ├── vid_0001/
│   │   └── <frame_index>.png
│   └── ...
├── shot_boundaries.csv
├── keyframes.csv
└── processing_log.csv
```

Keyframes are exported as PNG files using their source frame indices.

---

## ⚙️ Configuration

| Parameter | Value | Description |
|---|---|---|
| `ANALYSIS_SECONDS` | `2.0` | Analysis sampling interval |
| `WINDOW_SECONDS` | `45.0` | Temporal window duration |
| `DUPLICATE_THRESHOLD` | `3.0` | Near-duplicate filtering threshold |
| `SECOND_FRAME_THRESHOLD` | `12.0` | Additional frame selection threshold |

These values are experimental, not benchmark-optimized.

---

## ⚠️ Limitations

- TransNetV2 detects shot transitions, not whiteboard content changes.
- Temporal sampling may miss short-lived content.
- The keyframe selection algorithm has not been fully optimized or benchmarked for extraction quality and computational efficiency.

---

## 🔗 References

- [TransNetV2 – Official Repository](https://github.com/soCzech/TransNetV2)
- [TransNetV2 Research Paper](https://arxiv.org/abs/2008.04838)
- **BOARD2VEC Issue:** [#23 – Detect Keyframes from Video Dataset](https://github.com/BlackJackMC/meeting-intelligence-assistant/issues/23)
- **Dataset:** Google Drive link to be added.
