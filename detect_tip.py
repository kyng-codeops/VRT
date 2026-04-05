#!/usr/bin/env python3
"""Detect telecine, interlace, or progressive content per scene.

Uses scenedetect to find scene boundaries, then runs ffmpeg's idet filter
on each scene to classify field structure.

Usage:
    python detect_tip.py -i input.mkv
    python detect_tip.py -i input.mkv --frames 100    # sample more frames per scene
"""

import argparse
import json
import re
import subprocess
import sys

import av
from scenedetect import detect, ContentDetector


def get_scene_timecodes(video_path):
    """Detect scene boundaries and return list of (start_sec, end_sec) tuples."""
    scene_list = detect(video_path, ContentDetector(), show_progress=True)
    scenes = []
    for scene in scene_list:
        start = scene[0].get_seconds()
        end = scene[1].get_seconds()
        scenes.append((start, end))
    return scenes


def analyze_scene_idet(video_path, start_sec, duration_sec):
    """Run ffmpeg idet filter on a segment and return detection counts.

    Returns dict with keys: tff, bff, progressive, undetermined
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "info",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", video_path,
        "-vf", "idet",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr

    # Parse idet summary from stderr
    # Example: [Parsed_idet_0 @ ...] Repeated Fields: Neither: 50 Top: 0 Bottom: 0
    # [Parsed_idet_0 @ ...] Single frame detection: TFF: 0 BFF: 0 Progressive: 48 Undetermined: 2
    counts = {"tff": 0, "bff": 0, "progressive": 0, "undetermined": 0}

    # Use the LAST "Single frame detection:" line — ffmpeg may emit multiple
    # idet instances (one from seeking, one from the actual decode segment).
    idet_line = None
    for line in stderr.split("\n"):
        if "Single frame detection:" in line:
            idet_line = line

    if idet_line:
        for token in ["TFF", "BFF", "Progressive", "Undetermined"]:
            idx = idet_line.find(f"{token}:")
            if idx >= 0:
                rest = idet_line[idx + len(token) + 1:].strip()
                num_str = rest.split()[0] if rest.split() else "0"
                try:
                    counts[token.lower()] = int(num_str)
                except ValueError:
                    pass

    return counts


def _parse_frame_count(stderr):
    """Parse frame count from ffmpeg's final progress/summary line."""
    for line in reversed(stderr.split("\n")):
        m = re.search(r'frame=\s*(\d+)', line)
        if m:
            return int(m.group(1))
    return 0


def check_hard_telecine(video_path, fps_val, duration_sec,
                        sample_duration=10.0):
    """Detect hard telecine (3:2 pulldown baked into progressive frames).

    MPEG-2 hard telecine uses repeat_first_field (RFF) flags to signal 3:2
    pulldown.  ffmpeg's decoder honours RFF and outputs only unique frames,
    so a 29.97fps container actually decodes ~23.976fps of unique frames.
    We detect this by comparing the container fps to the actual decoded
    frame rate over a sample window.

    Returns (is_hard_telecine, fps_ratio).
    """
    if not (29.9 < fps_val < 30.1):
        return False, 0.0

    # Sample from the middle of the video to avoid logos/credits
    sample_start = max(0, (duration_sec / 2) - (sample_duration / 2))
    if duration_sec > 0 and sample_start + sample_duration > duration_sec:
        sample_start = max(0, duration_sec - sample_duration)

    # Count actual decoded frames (ffmpeg removes RFF duplicates automatically)
    cmd = [
        "ffmpeg", "-hide_banner",
        "-ss", str(sample_start), "-t", str(sample_duration),
        "-i", video_path,
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    decoded_frames = _parse_frame_count(result.stderr)

    if decoded_frames == 0:
        return False, 0.0

    actual_fps = decoded_frames / sample_duration
    fps_ratio = fps_val / actual_fps

    # 3:2 pulldown: container 29.97 / decoded 23.976 = ratio ~1.25 (5:4)
    is_hard_telecine = 1.20 <= fps_ratio <= 1.30
    return is_hard_telecine, fps_ratio


def classify_scene(counts):
    """Classify a scene based on idet counts."""
    tff = counts["tff"]
    bff = counts["bff"]
    prog = counts["progressive"]
    undet = counts["undetermined"]
    total = tff + bff + prog

    if total == 0:
        if undet > 0:
            return "Undetermined"
        return "Unknown (no data)"

    interlaced = tff + bff
    interlaced_ratio = interlaced / total

    if interlaced_ratio < 0.05:
        return "Progressive"
    elif interlaced_ratio < 0.35:
        dominant = "TFF" if tff >= bff else "BFF"
        return f"Telecined ({dominant}, {interlaced_ratio:.0%} interlaced)"
    else:
        dominant = "TFF" if tff >= bff else "BFF"
        return f"Interlaced ({dominant}, {interlaced_ratio:.0%} interlaced)"


def quick_detect(video_path, sample_duration=30.0):
    """Fast whole-video TIP detection without scene segmentation.

    Runs idet on a sample from the middle of the video and checks for hard
    telecine. Returns a dict with:
        mode:        "telecine" | "interlaced" | "progressive"
        field_order: "tff" | "bff" | "unknown"
        detail:      human-readable explanation string
    """
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        fps = stream.average_rate or stream.guessed_rate
        fps_val = float(fps)
        duration_sec = float(container.duration / 1_000_000) if container.duration else 0

    # --- Hard telecine check (MPEG-2 RFF / 3:2 pulldown baked in) ---
    is_hard_tc, tc_fps_ratio = check_hard_telecine(video_path, fps_val, duration_sec)
    if is_hard_tc:
        actual_fps = fps_val / tc_fps_ratio
        return {
            "mode": "telecine",
            "field_order": "tff",  # hard TC doesn't have a meaningful field order
            "detail": f"Hard telecine (container {fps_val:.2f}fps, decoded {actual_fps:.2f}fps)",
        }

    # --- idet-based detection on a sample from the middle ---
    sample_start = max(0, (duration_sec / 2) - (sample_duration / 2))
    if duration_sec > 0 and sample_start + sample_duration > duration_sec:
        sample_start = max(0, duration_sec - sample_duration)

    counts = analyze_scene_idet(video_path, sample_start, sample_duration)
    tff = counts["tff"]
    bff = counts["bff"]
    prog = counts["progressive"]
    undet = counts["undetermined"]
    total = tff + bff + prog

    if total == 0:
        # If all frames undetermined, fall back to progressive
        return {
            "mode": "progressive",
            "field_order": "unknown",
            "detail": f"Undetermined ({undet} frames could not be classified)",
        }

    interlaced = tff + bff
    interlaced_ratio = interlaced / total
    dominant = "tff" if tff >= bff else "bff"

    if interlaced_ratio < 0.05:
        return {
            "mode": "progressive",
            "field_order": "unknown",
            "detail": f"Progressive ({prog}/{total} frames progressive)",
        }
    elif interlaced_ratio < 0.35:
        return {
            "mode": "telecine",
            "field_order": dominant,
            "detail": f"Soft telecine ({dominant.upper()}, {interlaced_ratio:.0%} interlaced)",
        }
    else:
        return {
            "mode": "interlaced",
            "field_order": dominant,
            "detail": f"Interlaced ({dominant.upper()}, {interlaced_ratio:.0%} interlaced)",
        }


def analyze_video(video_path, sample_frames=50):
    """Analyze video for telecine/interlace per scene."""
    # Get framerate and duration
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        fps = stream.average_rate or stream.guessed_rate
        fps_val = float(fps)
        duration_sec = float(container.duration / 1_000_000) if container.duration else 0

    sample_duration = sample_frames / fps_val

    print(f"Source: {video_path}")
    print(f"FPS: {fps} ({fps_val:.4f})")
    print(f"Duration: {duration_sec:.1f}s")
    print(f"Sampling {sample_frames} frames ({sample_duration:.2f}s) per scene\n")

    # Pre-check for hard telecine (3:2 pulldown baked into progressive frames)
    print("Checking for hard telecine...")
    is_hard_tc, tc_fps_ratio = check_hard_telecine(video_path, fps_val, duration_sec)
    if is_hard_tc:
        actual_fps = fps_val / tc_fps_ratio
        print(f"Hard telecine detected (container {fps_val:.2f}fps, decoded {actual_fps:.2f}fps, ratio {tc_fps_ratio:.3f})\n")
    else:
        print(f"No hard telecine pattern (fps ratio: {tc_fps_ratio:.3f})\n")

    print("Detecting scene boundaries...")
    scenes = get_scene_timecodes(video_path)
    print(f"Found {len(scenes)} scenes\n")

    # Summary counters (scene count and duration-weighted)
    summary = {"Progressive": 0, "Hard Telecine": 0, "Telecined": 0, "Interlaced": 0, "Undetermined": 0, "Unknown": 0}
    weighted = {"Progressive": 0.0, "Hard Telecine": 0.0, "Telecined": 0.0, "Interlaced": 0.0, "Undetermined": 0.0, "Unknown": 0.0}

    for i, (start, end) in enumerate(scenes):
        scene_dur = end - start
        dur = min(sample_duration, scene_dur)
        counts = analyze_scene_idet(video_path, start, dur)
        classification = classify_scene(counts)

        # Override progressive/undetermined → hard telecine when globally detected
        if is_hard_tc and classification in ("Progressive", "Undetermined"):
            classification = "Hard Telecine (3:2 pulldown)"

        print(f"Scene {i:4d} [{start:8.2f}s - {end:8.2f}s]: {classification}  "
              f"(TFF:{counts['tff']} BFF:{counts['bff']} Prog:{counts['progressive']} Undet:{counts['undetermined']})")

        # Bucket by label
        if "Hard Telecine" in classification:
            key = "Hard Telecine"
        elif "Progressive" in classification:
            key = "Progressive"
        elif "Telecined" in classification:
            key = "Telecined"
        elif "Interlaced" in classification:
            key = "Interlaced"
        elif "Undetermined" in classification:
            key = "Undetermined"
        else:
            key = "Unknown"
        summary[key] += 1
        weighted[key] += scene_dur

    # Print summary
    total = len(scenes)
    total_dur = sum(weighted.values())
    print(f"\n{'='*60}")
    print(f"Summary ({total} scenes, {total_dur:.1f}s total):")
    for label in summary:
        count = summary[label]
        pct = count / total * 100 if total else 0
        dur_pct = weighted[label] / total_dur * 100 if total_dur else 0
        print(f"  {label:15s}: {count:4d} scenes ({pct:5.1f}%)  {weighted[label]:8.1f}s ({dur_pct:5.1f}%)")

    # Overall verdict — use duration-weighted classification
    tc_dur = weighted["Hard Telecine"] + weighted["Telecined"]
    il_dur = weighted["Interlaced"]
    prog_dur = weighted["Progressive"]

    if tc_dur > il_dur and tc_dur > prog_dur:
        tc_type = "Hard telecine" if weighted["Hard Telecine"] > weighted["Telecined"] else "Telecined"
        print(f"\nVerdict: {tc_type} content — use IVTC (inverse telecine)")
    elif il_dur > prog_dur:
        print(f"\nVerdict: Interlaced content — use deinterlace (nnedi3)")
    elif tc_dur + il_dur > 0:
        print(f"\nVerdict: Mixed content (mostly progressive with some interlaced/telecine scenes)")
    else:
        print(f"\nVerdict: Progressive content — no deinterlacing needed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect telecine, interlace, or progressive per scene using idet + scenedetect",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Input video file")
    parser.add_argument("--frames", type=int, default=300,
                        help="Number of frames to sample per scene (default: 300)")
    args = parser.parse_args()
    analyze_video(args.input, sample_frames=args.frames)
