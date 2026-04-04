import vapoursynth as vs
import os
from vsrife import rife

core = vs.core
core.max_cache_size = 2048  # Limit VS frame cache to 2GB RAM

# Get input file from environment (set by run_fps_interpolate.py)
input_file = os.environ.get("VRT_INPUT")
if not input_file:
    raise vs.Error("VRT_INPUT environment variable not set. Use run_fps_interpolate.py to run this script.")

# Load your source
clip = core.ffms2.Source(input_file)

# Convert to RGBS (float32) for RIFE — RGBH (float16) also supported for speed
# BT.601 matrix for SD/DVD content; use "709" for HD content
clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="470bg")

# Get interpolation factor from environment (default: 2x)
factor = int(os.environ.get("VRT_FACTOR", "2"))

# RIFE frame interpolation
# model: "4.25" is the default high-quality model
#   - "4.25.heavy" for maximum quality (slower)
#   - "4.25.lite" for faster processing (slightly lower quality)
#   - "4.26.heavy" for latest heavy model
# ensemble: True for smoother results in uncertain areas (slower)
# sc: True to detect scene changes and avoid interpolating across cuts
# scale: optical flow resolution — use 0.5 for 4K, 1.0 for HD/SD
interpolated = rife(clip, model="4.25", factor_num=factor, factor_den=1,
                    ensemble=True, sc=True, sc_threshold=0.15,
                    scale=1.0, auto_download=True)

# To use RGBH (FP16) for faster inference with slightly lower precision:
#   clip = core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s="470bg")
# To use the heavy model for maximum quality:
#   interpolated = rife(clip, model="4.25.heavy", ...)

# Convert back to YUV for y4m output (444 10-bit to match FFV1 lossless output)
interpolated = core.resize.Bicubic(interpolated, format=vs.YUV444P10, matrix_s="470bg")

interpolated.set_output()
