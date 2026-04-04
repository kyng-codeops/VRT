#!/usr/bin/env python3
"""Run detelecine / deinterlace pipeline: vspipe → FIFO → ffmpeg.

Auto-detects telecine vs interlace and field order (TFF/BFF), then applies
the correct processing. Optionally upscales with RVRT VSR (4x) or a custom
Real-ESRGAN .pth model.

Usage:
    python run_detelecine.py -i input.mkv                        # auto-detect, FFV1 lossless
    python run_detelecine.py -i input.mkv --analyze              # analyze only, no output
    python run_detelecine.py -i input.mkv --mode ivtc             # force inverse telecine
    python run_detelecine.py -i input.mkv --mode deinterlace      # force deinterlace
    python run_detelecine.py -i input.mkv --field-order bff       # force bottom-field-first
    python run_detelecine.py -i input.mkv --upscale vsr           # 4x RVRT super-resolution
    python run_detelecine.py -i input.mkv --upscale esrgan --esrgan-model weights.pth
    python run_detelecine.py -i input.mkv --gpu --encoder hevc_nvenc --cq 18
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import av

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

VPY_SCRIPT = "detelecine.vpy"


def get_framerate(input_path: str) -> tuple[int, int]:
    """Get the frame rate from input video as (numerator, denominator)."""
    with av.open(input_path) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        return rate.numerator, rate.denominator


def get_resolution(input_path: str) -> tuple[int, int]:
    """Get width and height from input video."""
    with av.open(input_path) as container:
        stream = container.streams.video[0]
        return stream.width, stream.height


def run_analysis(input_path: str, env: dict) -> dict:
    """Run the .vpy in analysis-only mode to detect field order and telecine."""
    with tempfile.NamedTemporaryFile(suffix=".json", prefix="vrt_analysis_",
                                     delete=False) as f:
        analysis_path = f.name

    analysis_env = env.copy()
    analysis_env["VRT_ANALYZE_ONLY"] = "1"
    analysis_env["VRT_ANALYSIS_OUT"] = analysis_path

    try:
        result = subprocess.run(
            ["vspipe", "-c", "y4m", VPY_SCRIPT, "-", "-e", "0"],
            env=analysis_env, capture_output=True, text=True, timeout=120,
        )
        # Script raises SystemExit(0) after writing analysis, vspipe may
        # report a non-zero exit — that's expected
        if os.path.isfile(analysis_path):
            with open(analysis_path) as f:
                return json.load(f)
    except subprocess.TimeoutExpired:
        print("WARNING: analysis timed out, using defaults", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: analysis failed: {e}", file=sys.stderr)
    finally:
        try:
            os.unlink(analysis_path)
        except FileNotFoundError:
            pass

    return {"field_order": "tff", "field_type": "interlaced", "mode": "telecine"}


def compute_output_fps(src_fps_num: int, src_fps_den: int, mode: str) -> tuple[int, int]:
    """Compute the output framerate based on processing mode.

    IVTC: 30000/1001 → 24000/1001 (drops 1 in 5 frames)
    Deinterlace (same-rate): fps stays the same
    Progressive: fps stays the same
    """
    src_fps = src_fps_num / src_fps_den
    if mode == "telecine" and abs(src_fps - 29.97) < 0.5:
        # Standard NTSC telecine: 30000/1001 → 24000/1001
        return 24000, 1001
    elif mode == "telecine" and abs(src_fps - 25.0) < 0.5:
        # PAL telecine (rare): 25 → 20
        return 20, 1
    else:
        return src_fps_num, src_fps_den


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
        description="Detelecine / deinterlace pipeline: vspipe → ffmpeg",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Input video file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file prefix, .mkv extension forced\n"
                        "(default: <input>_ivtc or <input>_deinterlaced)")
    parser.add_argument("-cr", "--cframes", type=int, default=4,
                        help="vspipe concurrent frame requests (default: 4)")

    # Processing mode
    mode_group = parser.add_argument_group("processing mode")
    mode_group.add_argument("--mode", default="auto",
                            choices=["auto", "ivtc", "deinterlace"],
                            help="Processing mode (default: auto-detect).\n"
                            "  auto:         analyze input and choose\n"
                            "  ivtc:         force inverse telecine (3:2 pulldown removal)\n"
                            "  deinterlace:  force nnedi3 deinterlacing")
    mode_group.add_argument("--field-order", default="auto",
                            choices=["auto", "tff", "bff"],
                            help="Field order (default: auto-detect from source).\n"
                            "  tff: top field first (standard for most DVD)\n"
                            "  bff: bottom field first")
    mode_group.add_argument("--analyze", action="store_true",
                            help="Analyze input only (detect telecine/interlace), no output")

    # Upscale
    up_group = parser.add_argument_group("upscaling")
    up_group.add_argument("--upscale", default="none",
                          choices=["none", "vsr", "esrgan"],
                          help="Upscale method after detelecine/deinterlace.\n"
                          "  none:   no upscaling\n"
                          "  vsr:    RVRT video super-resolution (4x, GPU)\n"
                          "  esrgan: Real-ESRGAN with custom .pth weights (GPU)")
    up_group.add_argument("--esrgan-model", default="",
                          help="Path to Real-ESRGAN .pth weights file.\n"
                          "Required when --upscale esrgan is used.")
    up_group.add_argument("--width", type=int, default=0,
                          help="Output width (resize after upscale, 0=no resize)")
    up_group.add_argument("--height", type=int, default=0,
                          help="Output height (resize after upscale, 0=no resize)")

    # Encoding
    enc_group = parser.add_argument_group("encoding")
    enc_group.add_argument("--gpu-lossless", action="store_true",
                           help="NVENC GPU lossless encoding (444 10-bit)")
    enc_group.add_argument("--gpu", action="store_true",
                           help="Use NVENC GPU encoding instead of FFV1 CPU.\n"
                           "--gpu-lossless takes precedence if both set.")
    enc_group.add_argument("--encoder", default="hevc_nvenc",
                           choices=["hevc_nvenc", "av1_nvenc"],
                           help="NVENC encoder (default: hevc_nvenc)")
    enc_group.add_argument("--cq", type=int, default=18,
                           help="Constant QP for NVENC (lower=better, default: 18)")

    args = parser.parse_args()

    if args.upscale == "esrgan" and not args.esrgan_model:
        parser.error("--esrgan-model is required when --upscale esrgan is used")

    return args


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent

    # Resolve paths before chdir so they're relative to the user's CWD
    input_path = str(Path(args.input).resolve())
    if not Path(input_path).is_file():
        sys.exit(f"ERROR: input file not found: {input_path}")

    esrgan_model_path = ""
    if args.esrgan_model:
        esrgan_model_path = str(Path(args.esrgan_model).resolve())
        if not Path(esrgan_model_path).is_file():
            sys.exit(f"ERROR: ESRGAN model not found: {esrgan_model_path}")

    # chdir to script dir so vspipe can find the .vpy and models/
    os.chdir(script_dir)

    # Build environment for .vpy
    env = os.environ.copy()
    env["VRT_INPUT"] = input_path
    env["VRT_MODE"] = args.mode
    env["VRT_FIELD_ORDER"] = args.field_order
    env["VRT_UPSCALE"] = args.upscale
    env["VRT_ESRGAN_MODEL"] = esrgan_model_path
    env["VRT_WIDTH"] = str(args.width)
    env["VRT_HEIGHT"] = str(args.height)

    # Detect source properties
    src_fps_num, src_fps_den = get_framerate(input_path)
    src_fps = src_fps_num / src_fps_den
    src_w, src_h = get_resolution(input_path)

    print(f"Input:      {input_path}")
    print(f"Source:     {src_w}x{src_h} @ {src_fps_num}/{src_fps_den} ({src_fps:.4f} fps)")

    # Run analysis phase
    print("Analyzing field order and telecine pattern...")
    analysis = run_analysis(input_path, env)
    detected_mode = analysis.get("mode", "telecine")
    detected_order = analysis.get("field_order", "tff")
    field_type = analysis.get("field_type", "interlaced")

    mode_labels = {
        "telecine": "Telecine (3:2 pulldown) → IVTC",
        "interlaced": "True interlace → nnedi3 deinterlace",
        "progressive": "Progressive (no processing needed)",
    }
    print(f"Field order: {detected_order.upper()}")
    print(f"Detection:   {mode_labels.get(detected_mode, detected_mode)}")

    if args.analyze:
        print("\nAnalysis complete (--analyze mode, no output generated).")
        return

    # Compute output framerate
    out_fps_num, out_fps_den = compute_output_fps(src_fps_num, src_fps_den, detected_mode)
    out_fps = out_fps_num / out_fps_den
    fps_str = f"{out_fps_num}/{out_fps_den}"

    # Determine output name
    if args.output:
        output_path = str(Path(args.output).resolve() if not Path(args.output).is_absolute()
                         else Path(args.output))
        output_path = str(Path(output_path).with_suffix(".mkv"))
    else:
        suffix = "_ivtc" if detected_mode == "telecine" else "_deinterlaced"
        stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{stem}{suffix}.mkv")

    upscale_label = {"none": "none", "vsr": "RVRT 4x VSR", "esrgan": f"Real-ESRGAN ({Path(esrgan_model_path).name})"}
    print(f"Output FPS: {fps_str} ({out_fps:.4f})")
    print(f"Upscale:    {upscale_label.get(args.upscale, args.upscale)}")
    print(f"Output:     {output_path}")

    # Create named pipe
    fifo_fd, fifo_path = tempfile.mkstemp(suffix=".y4m", prefix="vrt_fifo_")
    os.close(fifo_fd)
    os.unlink(fifo_path)
    os.mkfifo(fifo_path)

    ffmpeg_proc = None
    vspipe_proc = None

    def cleanup(signum=None, frame=None):
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
