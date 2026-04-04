#!/usr/bin/env python3
"""Run frame rate interpolation pipeline: vspipe → FIFO → ffmpeg.

Doubles (or multiplies) the frame rate using RIFE AI interpolation.
Detects input framerate automatically and handles graceful shutdown
on Ctrl+C / SIGTERM, cleaning up both vspipe and ffmpeg subprocesses.

Usage:
    python run_fps_interpolate.py -i input.mp4                   # FFV1 lossless (CPU)
    python run_fps_interpolate.py -i input.mp4 -o out            # custom output prefix
    python run_fps_interpolate.py -i input.mp4 --gpu             # NVENC HEVC encoding
    python run_fps_interpolate.py -i input.mp4 --gpu-lossless    # NVENC AV1 lossless 444 10-bit
"""

import argparse
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import av

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

VPY_SCRIPT = "fps_interpolate.vpy"


def get_framerate(input_path: str) -> tuple[int, int]:
    """Get the frame rate from input video as (numerator, denominator)."""
    with av.open(input_path) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        return rate.numerator, rate.denominator


def build_ffmpeg_cmd(fifo_path, input_path, output_path, fps_str, args):
    """Build the ffmpeg command based on encoding mode."""
    cmd = [
        "ffmpeg", "-hide_banner", "-stats",
        "-f", "yuv4mpegpipe", "-i", fifo_path,
        "-i", input_path,
        "-map", "0:v", "-map", "1:a", "-map_metadata", "1",
        "-r", fps_str,
    ]

    if args.gpu or args.gpu_lossless:
        if args.gpu_lossless:
            # AV1 NVENC for true lossless 444 10-bit
            cmd += [
                "-c:v", "av1_nvenc",
                "-preset", "p1",
                "-tune", "lossless",
                "-rc", "constqp",
                "-qp", "0",
                "-b:v", "0",
                "-pix_fmt", "yuv444p10le",
            ]
            print("Encoding: av1_nvenc (GPU), lossless 444 10-bit")
        else:
            encoder = args.encoder
            cmd += [
                "-c:v", encoder,
                "-preset", "p7",
                "-tune", "hq",
                "-rc", "constqp",
                "-qp", str(args.cq),
                "-b:v", "0",
                "-pix_fmt", "yuv444p10le",
            ]
            print(f"Encoding: {encoder} (GPU), QP={args.cq}")
    else:
        cmd += [
            "-c:v", "ffv1", "-level", "3", "-slicecrc", "1",
            "-slices", "12", "-threads", "12",
            "-pix_fmt", "yuv444p10le",
        ]
        print("Encoding: FFV1 lossless (CPU, 12 slices/threads)")

    cmd += ["-c:a", "copy", "-shortest", output_path, "-y"]
    return cmd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frame rate interpolation pipeline: vspipe → ffmpeg",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Input video file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file prefix, .mkv extension forced (default: <input>_interpolated)")
    parser.add_argument("-cr", "--cframes", type=int, default=4,
                        help="vspipe concurrent frame requests (default: 4)")
    parser.add_argument("--factor", type=int, default=2,
                        help="Frame rate multiplier (default: 2 = double)")
    parser.add_argument("--gpu-lossless", action="store_true",
                        help="NVENC GPU lossless encoding (444 10-bit, good for intermediates)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use custom NVENC GPU encoding instead of gpu-lossless or FFV1 CPU.\n"
                        "Cannot be used with --gpu-lossless; if both are set, --gpu-lossless\n"
                        "takes precedence.")
    parser.add_argument("--encoder", default="hevc_nvenc",
                        choices=["hevc_nvenc", "av1_nvenc"],
                        help="NVENC encoder to use with --gpu (default: hevc_nvenc).\n"
                        "Requires --gpu and is ignored if --gpu-lossless is set.")
    parser.add_argument("--cq", type=int, default=18,
                        help="Constant QP for NVENC (lower=better, default: 18).\n"
                        "Requires --gpu and is ignored if --gpu-lossless is set.")
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    input_path = str(Path(args.input).resolve())
    if not Path(input_path).is_file():
        sys.exit(f"ERROR: input file not found: {input_path}")

    if args.output:
        output_path = str(Path(args.output).resolve().with_suffix(".mkv"))
    else:
        stem = Path(input_path).stem
        output_path = str(script_dir / f"{stem}_interpolated.mkv")

    # Detect framerate and compute output rate
    fps_num, fps_den = get_framerate(input_path)
    src_fps = fps_num / fps_den
    out_fps_num = fps_num * args.factor
    out_fps = out_fps_num / fps_den
    fps_str = f"{out_fps_num}/{fps_den}"

    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print(f"Source FPS: {fps_num}/{fps_den} ({src_fps:.4f})")
    print(f"Output FPS: {fps_str} ({out_fps:.4f}) [{args.factor}x]")

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
        if vspipe_proc and vspipe_proc.poll() is None:
            print(f"\nKilling vspipe (pid {vspipe_proc.pid})...")
            vspipe_proc.kill()
            vspipe_proc.wait()

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

    # Pass input file and factor to .vpy via environment
    env = os.environ.copy()
    env["VRT_INPUT"] = input_path
    env["VRT_FACTOR"] = str(args.factor)

    try:
        ffmpeg_cmd = build_ffmpeg_cmd(fifo_path, input_path, output_path, fps_str, args)
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd)

        vspipe_proc = subprocess.Popen([
            "vspipe", "-c", "y4m", "-p",
            VPY_SCRIPT, fifo_path,
            "-r", str(args.cframes),
        ], env=env)

        vspipe_proc.wait()
        vspipe_exit = vspipe_proc.returncode

        ffmpeg_proc.wait()
        ffmpeg_exit = ffmpeg_proc.returncode

    except Exception:
        cleanup()
        raise

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
