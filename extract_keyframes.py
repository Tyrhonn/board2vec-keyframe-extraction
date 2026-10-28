
"""Export representative keyframes from TransNetV2 shots."""

from pathlib import Path
import csv
import cv2
import numpy as np

VIDEO_DIR = Path("dataset/videos")
OUTPUT_DIR = Path("dataset/keyframes")
SHOT_CSV = Path("dataset/shot_boundaries.csv")
KEYFRAME_CSV = Path("dataset/keyframes.csv")
LOG_CSV = Path("dataset/processing_log.csv")

# Sampling for analysis only, NOT fixed keyframe export.
ANALYSIS_SECONDS = 2.0
WINDOW_SECONDS = 45.0

# Experimental grayscale difference thresholds (0-255).
DUPLICATE_THRESHOLD = 3.0
SECOND_FRAME_THRESHOLD = 12.0


def load_shots(path):
    shots = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            start, end = map(int, line.split())
            if start < 0 or end < start:
                raise ValueError(f"Invalid shot: {line.strip()}")
            shots.append((start, end))
    if not shots:
        raise ValueError("No shots found")
    return shots


def descriptor(frame):
    """Low-resolution grayscale representation for comparison."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (96, 54))
    return small.astype(np.float32)


def distance(a, b):
    return float(np.mean(np.abs(a - b)))


def select_representatives(samples):
    """Select a medoid and optionally one visually different frame."""
    if len(samples) == 1:
        return samples

    features = [item[1] for item in samples]
    n = len(features)
    distances = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            d = distance(features[i], features[j])
            distances[i, j] = d
            distances[j, i] = d

    # Medoid: smallest total distance to other samples.
    main_idx = int(np.argmin(distances.sum(axis=1)))
    selected = [samples[main_idx]]

    # Keep one additional candidate if sufficiently different.
    farthest_idx = int(np.argmax(distances[main_idx]))
    if distances[main_idx, farthest_idx] >= SECOND_FRAME_THRESHOLD:
        selected.append(samples[farthest_idx])

    return selected


def read_frame(capture, frame_id):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Cannot decode frame {frame_id}")
    return frame


def process_video(video_path, shot_writer, key_writer):
    video_id = video_path.stem
    scenes_path = Path(str(video_path) + ".scenes.txt")
    if not scenes_path.exists():
        raise FileNotFoundError(f"Missing {scenes_path}")

    shots = load_shots(scenes_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    exported = set()
    last_selected_feature = None
    output_dir = OUTPUT_DIR / video_id

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if not np.isfinite(fps) or fps <= 0 or total <= 0:
            raise RuntimeError("Invalid FPS or frame count")

        analysis_step = max(1, round(fps * ANALYSIS_SECONDS))
        window_size = max(1, round(fps * WINDOW_SECONDS))

        for shot_id, (start, end) in enumerate(shots, start=1):
            if end >= total:
                raise ValueError(f"Shot {shot_id} exceeds video length")

            shot_writer.writerow([
                video_id, shot_id, start, end,
                f"{start / fps:.3f}", f"{end / fps:.3f}"
            ])

            # Partition long shots into temporal analysis windows.
            for window_start in range(start, end + 1, window_size):
                window_end = min(end, window_start + window_size - 1)

                frame_ids = list(range(
                    window_start, window_end + 1, analysis_step
                ))
                frame_ids.append((window_start + window_end) // 2)
                frame_ids = sorted(set(frame_ids))

                samples = []
                for frame_id in frame_ids:
                    frame = read_frame(capture, frame_id)
                    samples.append((frame_id, descriptor(frame)))

                chosen = select_representatives(samples)

                for frame_id, feature in sorted(chosen):
                    if frame_id in exported:
                        continue

                    if last_selected_feature is not None:
                        diff = distance(feature, last_selected_feature)
                        if diff < DUPLICATE_THRESHOLD:
                            continue
                    else:
                        diff = 0.0

                    # Re-read the ORIGINAL frame to export full resolution.
                    frame = read_frame(capture, frame_id)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    image_path = output_dir / f"{frame_id:05d}.png"

                    if not cv2.imwrite(str(image_path), frame):
                        raise RuntimeError(f"Cannot save {image_path}")

                    # Verify the exported PNG and dimensions.
                    check = cv2.imread(str(image_path))
                    if check is None or check.shape[:2] != frame.shape[:2]:
                        raise RuntimeError(f"Invalid exported PNG: {image_path}")

                    height, width = frame.shape[:2]
                    key_writer.writerow([
                        video_id, shot_id, frame_id,
                        f"{frame_id / fps:.3f}",
                        width, height, f"{diff:.3f}",
                        str(image_path)
                    ])

                    exported.add(frame_id)
                    last_selected_feature = feature

    finally:
        capture.release()

    return len(exported), len(shots)



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(VIDEO_DIR.glob("vid_*.mp4"))

    if not videos:
        print("No videos to process")
        return

    completed = set()

    if LOG_CSV.exists():
        with LOG_CSV.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["status"] == "success":
                    completed.add(row["video"])

    with (
        SHOT_CSV.open("a", newline="", encoding="utf-8") as shots_file,
        KEYFRAME_CSV.open("a", newline="", encoding="utf-8") as keys_file,
        LOG_CSV.open("a", newline="", encoding="utf-8") as log_file,
    ):
        shot_writer = csv.writer(shots_file)
        key_writer = csv.writer(keys_file)
        log_writer = csv.writer(log_file)

        if SHOT_CSV.stat().st_size == 0:
            shot_writer.writerow([
                "video_id", "shot_id", "start_frame", "end_frame",
                "start_seconds", "end_seconds"
            ])

        if KEYFRAME_CSV.stat().st_size == 0:
            key_writer.writerow([
                "video_id", "shot_id", "frame_id",
                "timestamp_seconds", "width", "height",
                "difference", "path"
            ])

        if LOG_CSV.stat().st_size == 0:
            log_writer.writerow([
                "video", "status", "shots", "keyframes", "reason"
            ])

        for video_path in videos:
            # Skip successfully processed videos
            if video_path.name in completed:
                print(f"SKIP: {video_path.name} (already processed)")
                continue

            # Check shot boundary file
            scene_path = Path(str(video_path) + ".scenes.txt")

            if not scene_path.is_file():
                reason = f"Missing shot boundary file: {scene_path.name}"

                log_writer.writerow([
                    video_path.name, "failed", "", "", reason
                ])

                print(f"{video_path.name}: FAILED - {reason}")
                continue

            output_dir = OUTPUT_DIR / video_path.stem

            if output_dir.exists() and any(output_dir.glob("*.png")):
                print(
                    f"REVIEW: {video_path.name} already has PNG files "
                    "but no successful log record. Skipping to avoid mixing results."
                )
                continue

            try:
                count, n_shots = process_video(
                    video_path, shot_writer, key_writer
                )

                log_writer.writerow([
                    video_path.name, "success", n_shots, count, ""
                ])

                print(
                    f"{video_path.name}: "
                    f"{n_shots} shots, {count} keyframes"
                )

            except (OSError, ValueError, RuntimeError, cv2.error) as exc:
                log_writer.writerow([
                    video_path.name, "failed", "", "", str(exc)
                ])
                print(f"{video_path.name}: FAILED - {exc}")


if __name__ == "__main__":
    main()
