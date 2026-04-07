import vapoursynth as vs
import os
from vsrife import rife

core = vs.core
core.max_cache_size = 2048  # Limit VS frame cache to 2GB RAM

# ---------------------------------------------------------------------------
# Helper: auto-detect color matrix from frame props or resolution
# ---------------------------------------------------------------------------
def get_matrix(clip):
    """Return the appropriate color matrix name for the clip.

    Checks _Matrix frame property first (most reliable), then falls back
    to resolution-based heuristics: HD (>=720p) uses BT.709, SD uses BT.601.
    """
    try:
        props = clip.get_frame(0).props
        matrix = props.get("_Matrix", None)
        if matrix == 1:
            return "709"
        if matrix in (5, 6):  # 5=BT.470BG, 6=SMPTE 170M
            return "470bg"
        if matrix == 9:
            return "2020ncl"
    except Exception:
        pass
    # Fallback to resolution heuristic
    if clip.width >= 1280 or clip.height >= 720:
        return "709"
    return "470bg"

# Get input file from environment (set by run_fps_interpolate.py)
input_file = os.environ.get("VRT_INPUT")
if not input_file:
    raise vs.Error("VRT_INPUT environment variable not set. Use run_fps_interpolate.py to run this script.")

# Load your source
clip = core.ffms2.Source(input_file)
_matrix = get_matrix(clip)  # Detect matrix from source before any processing

# Convert to RGBS (float32) for RIFE — RGBH (float16) also supported for speed
clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s=_matrix)

# Get interpolation factor from environment (default: 2x)
factor = int(os.environ.get("VRT_FACTOR", "2"))

# RIFE frame interpolation
# model: "4.25" is the default high-quality model
#   - "4.25.heavy" for maximum quality (slower)
#   - "4.25.lite" for faster processing (slightly lower quality)
#   - "4.26.heavy" for latest heavy model
# ensemble: True for smoother results in uncertain areas (slower)
#     Not supported by all models (e.g. v4.25 does not support it).
# sc: scene change detection — requires the VapourSynth MiscFilters plugin
#     (misc.SCDetect). Set to False if MiscFilters is not installed.
# scale: optical flow resolution — use 0.5 for 4K, 1.0 for HD/SD
try:
    core.misc  # check if MiscFilters plugin is available
    sc_enabled = True
except AttributeError:
    sc_enabled = False

interpolated = rife(clip, model="4.25", factor_num=factor, factor_den=1,
                    ensemble=False, sc=sc_enabled, sc_threshold=0.15,
                    scale=1.0, auto_download=True)

# To use RGBH (FP16) for faster inference with slightly lower precision:
#   clip = core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s=_matrix)
# To use the heavy model for maximum quality:
#   interpolated = rife(clip, model="4.25.heavy", ...)

# Convert back to YUV for y4m output (444 10-bit to match FFV1 lossless output)
interpolated = core.resize.Bicubic(interpolated, format=vs.YUV444P10, matrix_s=_matrix)

interpolated.set_output()
