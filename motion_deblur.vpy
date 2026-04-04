import vapoursynth as vs
import os
from vsrvrt import Deblur

core = vs.core
core.max_cache_size = 2048  # Limit VS frame cache to 2GB RAM

# Get input file from environment (set by run_motion_deblur.py)
input_file = os.environ.get("VRT_INPUT")
if not input_file:
    raise vs.Error("VRT_INPUT environment variable not set. Use run_motion_deblur.py to run this script.")

# Load your source
clip = core.ffms2.Source(input_file)

# Convert to RGB for Deblur (BT.601 for SD/DVD content)
clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="470bg")

# Remove blur using the GoPro deblurring model (general purpose)
# tile_size: (temporal, height, width) — 0 means no tiling on that axis
#   Larger temporal tile = more context = better quality but more VRAM
# tile_overlap: higher values reduce tile boundary artifacts
# use_fp16=False for full precision (2x VRAM cost, marginal quality gain)
deblurred = Deblur(clip, model="gopro", use_fp16=True,
                   tile_size=(16, 0, 0), tile_overlap=(4, 0, 0),
                   chunk_overlap=24, preview_mode=True)

# --- Motion-adaptive blending ---
# Only apply deblurring to areas with motion; preserve original in static regions
# to avoid removing intentional depth-of-field / background blur.

# Work in GRAY for motion detection (luma channel from YUV)
gray = core.resize.Bicubic(clip, format=vs.GRAY8, matrix_s="470bg")

# Inter-frame difference as a motion proxy
prev_frame = core.std.DuplicateFrames(gray, [0])           # shift forward by 1
prev_frame = core.std.DeleteFrames(prev_frame, [gray.num_frames])
next_frame = core.std.DeleteFrames(gray, [0])
next_frame = core.std.DuplicateFrames(next_frame, [next_frame.num_frames - 1])

# Combine forward and backward differences for robustness
diff_fwd = core.std.Expr([gray, prev_frame], ["x y - abs"])
diff_bwd = core.std.Expr([gray, next_frame], ["x y - abs"])
motion = core.std.Expr([diff_fwd, diff_bwd], ["x y max"])

# Smooth the mask spatially (blur + threshold)
# Large kernel to create soft regions rather than pixel-level noise
motion = core.std.Convolution(motion, [1]*9, mode="s")      # 3x3 box blur
motion = core.std.Convolution(motion, [1]*9, mode="s")      # apply twice for 5x5-ish

# Expand and soften: threshold low to catch subtle motion, then clamp
# threshold=8/255: pixels with >8 levels of inter-frame change count as motion
# Scale to full range for blending mask: 0=static (keep original), 1=motion (use deblurred)
motion = core.std.Expr([motion], ["x 8 > x 4 * 0 ? 255 min"])

# Another blur pass for soft mask edges
motion = core.std.Convolution(motion, [1]*9, mode="s")
motion = core.std.Convolution(motion, [1]*9, mode="s")

# Convert mask to match output format (RGBS, 3 planes)
mask = core.resize.Point(motion, format=vs.GRAYS)

# Blend: deblurred where motion is high, original where static
result = core.std.MaskedMerge(clip, deblurred, mask)

# Convert back to YUV for y4m output (444 to match FFV1 lossless output)
result = core.resize.Bicubic(result, format=vs.YUV444P10, matrix_s="470bg")

result.set_output()

# To disable the motion mask and apply deblurring uniformly to all areas,
# replace the "result" lines above with:
#   result = core.resize.Bicubic(deblurred, format=vs.YUV444P10, matrix_s="470bg")
#   result.set_output()
# The motion mask code (lines 26-58) can be left in place — it will be
# optimized away by VapourSynth since nothing references its output.
