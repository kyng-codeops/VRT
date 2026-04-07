#!/usr/bin/env python3
"""Run motion deblurring pipeline: vspipe → FIFO → ffmpeg.

Detects input framerate automatically and handles graceful shutdown
on Ctrl+C / SIGTERM, cleaning up both vspipe and ffmpeg subprocesses.

Usage:
    python run_motion_deblur.py -i input.mp4                  # FFV1 lossless (CPU)
    python run_motion_deblur.py -i input.mp4 -o out.mkv       # custom output name
    python run_motion_deblur.py -i input.mp4 --gpu             # NVENC HEVC encoding
    python run_motion_deblur.py -i input.mp4 --gpu --encoder hevc_nvenc --cq 18
    python run_motion_deblur.py -i input.mp4 --gpu-lossless    # NVENC lossless 444 10-bit
    python run_motion_deblur.py -i input.mp4 --cpu-hevc --cq 20 --x265-preset slow
    python run_motion_deblur.py -i input.mp4 --upscale esrgan --esrgan-model weights.pth
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

VPY_SCRIPT = "motion_deblur.vpy"


def get_framerate(input_path: str) -> str:
    """Get the frame rate from input video as a ratio string (e.g. '30000/1001')."""
    with av.open(input_path) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        return f"{rate.numerator}/{rate.denominator}"


def build_ffmpeg_cmd(fifo_path, input_path, output_path, fps, args):
    """Build the ffmpeg command based on encoding mode."""
    cmd = [
        "ffmpeg", "-hide_banner", "-stats",
        "-f", "yuv4mpegpipe", "-i", fifo_path,
        "-i", input_path,
        "-map", "0:v", "-map", "1:a", "-map_metadata", "1",
        "-r", fps,
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
    elif args.cpu_hevc:
        cmd += [
            "-c:v", "libx265",
            "-preset", args.x265_preset,
            "-crf", str(args.cq),
            "-pix_fmt", "yuv444p10le",
        ]
        print(f"Encoding: libx265 (CPU), preset={args.x265_preset}, "
              f"CRF={args.cq}, 444 10-bit")
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
        description="Motion deblur pipeline: vspipe → ffmpeg",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Input video file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file prefix, .mkv extension forced (default: <input>_deblurred)")
    parser.add_argument("-cr", "--cframes", type=int, default=4,
                        help="vspipe concurrent frame requests (default: 4)")
    parser.add_argument("--gpu-lossless", action="store_true",
                        help="NVENC GPU lossless encoding (444 10-bit)")
    parser.add_argument("--gpu", action="store_true",
                        help="NVENC GPU lossy encoding")
    parser.add_argument("--cpu-hevc", action="store_true",
                        help="CPU libx265 HEVC encoding (444 10-bit)")
    parser.add_argument("--x265-preset", default="medium",
                        choices=["ultrafast", "superfast", "veryfast", "faster",
                                 "fast", "medium", "slow", "slower", "veryslow"],
                        help="libx265 preset (default: medium)")
    parser.add_argument("--encoder", default="hevc_nvenc",
                        choices=["hevc_nvenc", "av1_nvenc"],
                        help="NVENC encoder for --gpu (default: hevc_nvenc)")
    parser.add_argument("--cq", type=int, default=18,
                        help="Quality value: CRF for --cpu-hevc, QP for --gpu")

    # Upscaling options
    up_group = parser.add_argument_group("Upscaling")
    up_group.add_argument("--upscale", choices=["none", "esrgan"], default="none",
                          help="Upscale after deblurring (default: none)")
    up_group.add_argument("--esrgan-model", default="",
                          help="Path to ESRGAN .pth model (required for --upscale esrgan)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate upscale options
    if args.upscale == "esrgan" and not args.esrgan_model:
        sys.exit("ERROR: --esrgan-model is required when --upscale esrgan is used")

    script_dir = Path(__file__).resolve().parent

    # Resolve paths before chdir so they're relative to the user's CWD
    input_path = str(Path(args.input).resolve())
    if not Path(input_path).is_file():
        sys.exit(f"ERROR: input file not found: {input_path}")

    if args.output:
        output_path = str(Path(args.output).resolve().with_suffix(".mkv"))
    else:
        stem = Path(input_path).stem
        output_path = str(script_dir / f"{stem}_deblurred.mkv")

    # chdir to script dir so vspipe can find the .vpy
    os.chdir(script_dir)

    # Detect framerate
    fps = get_framerate(input_path)
    print(f"Input:     {input_path}")
    print(f"Output:    {output_path}")
    num, den = fps.split("/")
    print(f"Framerate: {fps} ({int(num)/int(den):.4f} fps)")

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

    # Pass input file to .vpy via environment variable
    env = os.environ.copy()
    env["VRT_INPUT"] = input_path
    env["VRT_UPSCALE"] = args.upscale
    if args.esrgan_model:
        esrgan_model_path = str(Path(args.esrgan_model).resolve())
        if not Path(esrgan_model_path).is_file():
            sys.exit(f"ERROR: ESRGAN model not found: {esrgan_model_path}")
        env["VRT_ESRGAN_MODEL"] = esrgan_model_path

    try:
        ffmpeg_cmd = build_ffmpeg_cmd(fifo_path, input_path, output_path, fps, args)
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
