#!/usr/bin/env python3
"""Run frame rate interpolation pipeline: vspipe → FIFO → ffmpeg.

Doubles (or multiplies) the frame rate using RIFE AI interpolation.
Detects input framerate automatically and handles graceful shutdown
on Ctrl+C / SIGTERM, cleaning up both vspipe and ffmpeg subprocesses.

Usage:
    python run_fps_interpolate.py -i input.mp4                   # FFV1 lossless (CPU)
    python run_fps_interpolate.py -i input.mp4 -o out            # custom output prefix
    python run_fps_interpolate.py -i input.mp4 --gpu             # NVENC HEVC encoding
    python run_fps_interpolate.py -i input.mp4 --gpu-lossless    # NVENC H.264 lossless 444 8-bit
    python run_fps_interpolate.py -i input.mp4 --cpu-hevc --cq 20  # libx265 compatible
    python run_fps_interpolate.py -i input.mp4 --dar correct     # resize to square pixels"""

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


def get_sar(input_path: str) -> tuple[int, int]:
    """Get sample aspect ratio (SAR) from input video as (num, den).

    Returns (1, 1) for square pixels or if SAR is not set.
    """
    with av.open(input_path) as container:
        stream = container.streams.video[0]
        sar = stream.sample_aspect_ratio
        if sar is None or sar == 0:
            return 1, 1
        return sar.numerator, sar.denominator


def build_ffmpeg_cmd(fifo_path, input_path, output_path, fps_str, args,
                     sar_num=1, sar_den=1):
    """Build the ffmpeg command based on encoding mode."""
    cmd = [
        "ffmpeg", "-hide_banner", "-stats",
        "-f", "yuv4mpegpipe", "-i", fifo_path,
        "-i", input_path,
        "-map", "0:v", "-map", "1:a?", "-map_metadata", "1",
        "-r", fps_str,
    ]

    if args.gpu or args.gpu_lossless:
        if args.gpu_lossless:
            cmd += [
                "-c:v", "h264_nvenc",
                "-preset", "p1",
                "-tune", "lossless",
                "-pix_fmt", "yuv444p",
            ]
            print("Encoding: h264_nvenc (GPU), lossless 444 8-bit")
        else:
            encoder = args.encoder
            cmd += [
                "-c:v", encoder,
                "-preset", "p7",
                "-tune", "hq",
                "-rc", "constqp",
                "-qp", str(args.cq),
                "-b:v", "0",
                "-pix_fmt", "yuv420p10le",
            ]
            print(f"Encoding: {encoder} (GPU), QP={args.cq}, 420 10-bit")
    elif args.cpu_hevc:
        cmd += [
            "-c:v", "libx265",
            "-preset", args.x265_preset,
            "-crf", str(args.cq),
            "-pix_fmt", "yuv420p10le",
        ]
        print(f"Encoding: libx265 (CPU), preset={args.x265_preset}, "
              f"CRF={args.cq}, 420 10-bit")
    else:
        cmd += [
            "-c:v", "ffv1", "-level", "3", "-slicecrc", "1",
            "-slices", "12", "-threads", "12",
            "-pix_fmt", "yuv444p10le",
        ]
        print("Encoding: FFV1 lossless (CPU, 12 slices/threads)")

    # For --dar copy, embed SAR metadata so players display correct aspect ratio
    if args.dar == "copy" and (sar_num != 1 or sar_den != 1):
        cmd += ["-vf", f"setsar={sar_num}/{sar_den}"]
        print(f"DAR:       copy SAR {sar_num}:{sar_den} to output")

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
    parser.add_argument("--cpu-hevc", action="store_true",
                        help="CPU libx265 HEVC encoding (420 10-bit, compatible)")
    parser.add_argument("--x265-preset", default="medium",
                        choices=["ultrafast", "superfast", "veryfast", "faster",
                                 "fast", "medium", "slow", "slower", "veryslow"],
                        help="libx265 preset (default: medium)")
    parser.add_argument("--encoder", default="hevc_nvenc",
                        choices=["hevc_nvenc", "av1_nvenc"],
                        help="NVENC encoder to use with --gpu (default: hevc_nvenc).\n"
                        "Requires --gpu and is ignored if --gpu-lossless is set.")
    parser.add_argument("--cq", type=int, default=18,
                        help="Quality value: CRF for --cpu-hevc, QP for --gpu (default: 18)")
    parser.add_argument("--dar", default="none",
                        choices=["none", "copy", "correct"],
                        help="Display aspect ratio handling (default: none).\n"
                        "  none:    ignore source DAR\n"
                        "  copy:    copy source SAR metadata to output\n"
                        "  correct: resize to square pixels")
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent

    # Resolve paths before chdir so they're relative to the user's CWD
    input_path = str(Path(args.input).resolve())
    if not Path(input_path).is_file():
        sys.exit(f"ERROR: input file not found: {input_path}")

    if args.output:
        output_path = str(Path(args.output).resolve().with_suffix(".mkv"))
    else:
        stem = Path(input_path).stem
        output_path = str(script_dir / f"{stem}_interpolated.mkv")

    # chdir to script dir so vspipe can find the .vpy
    os.chdir(script_dir)

    # Detect framerate and compute output rate
    fps_num, fps_den = get_framerate(input_path)
    sar_num, sar_den = get_sar(input_path)
    sar_is_nonsquare = (sar_num != sar_den)
    sar_str = f" SAR {sar_num}:{sar_den}" if sar_is_nonsquare else ""

    src_fps = fps_num / fps_den
    out_fps_num = fps_num * args.factor
    out_fps = out_fps_num / fps_den
    fps_str = f"{out_fps_num}/{fps_den}"

    print(f"Input:      {input_path}{sar_str}")
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
        """Stop subprocesses and remove FIFO on exit."""
        # Kill vspipe first (stop producing frames) then let ffmpeg finalize
        if vspipe_proc and vspipe_proc.poll() is None:
            print(f"\nKilling vspipe (pid {vspipe_proc.pid})...")
            vspipe_proc.kill()
            try:
                vspipe_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if ffmpeg_proc and ffmpeg_proc.poll() is None:
            print(f"Stopping ffmpeg (pid {ffmpeg_proc.pid}), finalizing output...")
            try:
                ffmpeg_proc.send_signal(signal.SIGINT)
                ffmpeg_proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, OSError):
                print(f"Force-killing ffmpeg (pid {ffmpeg_proc.pid})...")
                ffmpeg_proc.kill()
                try:
                    ffmpeg_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
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

    # Pass DAR correction info to .vpy for --dar correct
    if args.dar == "correct" and sar_is_nonsquare:
        env["VRT_DAR_CORRECT"] = "1"
        env["VRT_SAR_NUM"] = str(sar_num)
        env["VRT_SAR_DEN"] = str(sar_den)

    try:
        ffmpeg_cmd = build_ffmpeg_cmd(
            fifo_path, input_path, output_path, fps_str, args,
            sar_num=sar_num, sar_den=sar_den
        )
        # start_new_session=True puts children in their own process group
        # so terminal Ctrl+C only goes to Python, giving us full cleanup control
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, start_new_session=True)

        vspipe_proc = subprocess.Popen([
            "vspipe", "-c", "y4m", "-p",
            VPY_SCRIPT, fifo_path,
            "-r", str(args.cframes),
        ], env=env, start_new_session=True)

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
