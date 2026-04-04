#!/usr/bin/env python3
"""Run motion deblurring pipeline: vspipe → FIFO → ffmpeg.

Detects input framerate automatically and handles graceful shutdown
on Ctrl+C / SIGTERM, cleaning up both vspipe and ffmpeg subprocesses.
"""

import os
import signal
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

INPUT_FILE = "blurry_video.mp4"
OUTPUT_FILE = "output.mkv"
VPY_SCRIPT = "motion_deblur.vpy"


def get_framerate(input_path: str) -> str:
    """Get the frame rate from input video as a ratio string (e.g. '30000/1001')."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "csv=p=0",
            input_path,
        ],
        capture_output=True, text=True, check=True,
    )
    fps_str = result.stdout.strip()
    # Validate it parses as a fraction
    Fraction(fps_str)
    return fps_str


def main():
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    input_path = str(script_dir / INPUT_FILE)
    output_path = str(script_dir / OUTPUT_FILE)

    # Detect framerate
    fps = get_framerate(input_path)
    print(f"Detected framerate: {fps} ({float(Fraction(fps)):.4f} fps)")

    # Create named pipe
    fifo_fd, fifo_path = tempfile.mkstemp(suffix=".y4m", prefix="vrt_fifo_")
    os.close(fifo_fd)
    os.unlink(fifo_path)
    os.mkfifo(fifo_path)

    ffmpeg_proc = None
    vspipe_proc = None

    def cleanup(signum=None, frame=None):
        """Stop subprocesses and remove FIFO on exit.

        Kills vspipe first to stop frame production, then sends SIGINT
        to ffmpeg so it finalizes the container (writes MKV trailer)
        and produces a playable partial output file.
        """
        # Stop vspipe immediately — no need for graceful shutdown
        if vspipe_proc and vspipe_proc.poll() is None:
            print(f"\nKilling vspipe (pid {vspipe_proc.pid})...")
            vspipe_proc.kill()
            vspipe_proc.wait()

        # Send SIGINT to ffmpeg so it flushes and writes the container trailer
        if ffmpeg_proc and ffmpeg_proc.poll() is None:
            print(f"Stopping ffmpeg (pid {ffmpeg_proc.pid}), finalizing output...")
            ffmpeg_proc.send_signal(signal.SIGINT)
            try:
                ffmpeg_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                ffmpeg_proc.kill()
                ffmpeg_proc.wait()

        try:
            os.unlink(fifo_path)
        except FileNotFoundError:
            pass
        if signum is not None:
            sys.exit(1)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # Start ffmpeg reading from FIFO (blocks until writer opens it)
        ffmpeg_proc = subprocess.Popen([
            "ffmpeg", "-hide_banner",
            "-f", "yuv4mpegpipe", "-i", fifo_path,
            "-i", input_path,
            "-map", "0:v", "-map", "1:a", "-map_metadata", "1",
            "-r", fps,
            "-c:v", "ffv1", "-level", "3", "-slicecrc", "1",
            "-slices", "12", "-pix_fmt", "yuv444p10le",
            "-c:a", "copy", "-shortest", output_path, "-y",
        ])

        # Feed frames into the FIFO
        vspipe_proc = subprocess.Popen([
            "vspipe", "-c", "y4m", "-p", VPY_SCRIPT, fifo_path, "-r", "4",
        ])

        vspipe_proc.wait()
        vspipe_exit = vspipe_proc.returncode

        ffmpeg_proc.wait()
        ffmpeg_exit = ffmpeg_proc.returncode

    except Exception:
        cleanup()
        raise

    # Clean up FIFO
    try:
        os.unlink(fifo_path)
    except FileNotFoundError:
        pass

    if vspipe_exit != 0:
        print(f"ERROR: vspipe failed with exit code {vspipe_exit}", file=sys.stderr)
        sys.exit(1)
    if ffmpeg_exit != 0:
        print(f"ERROR: ffmpeg failed with exit code {ffmpeg_exit}", file=sys.stderr)
        sys.exit(1)

    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
