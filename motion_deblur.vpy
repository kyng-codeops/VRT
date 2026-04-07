import vapoursynth as vs
import os
import sys
from vsrvrt import Deblur

# Add script directory to path for local imports (models.network_rrdbnet)
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

core = vs.core
core.max_cache_size = 2048  # Limit VS frame cache to 2GB RAM

# Get input file from environment (set by run_motion_deblur.py)
input_file = os.environ.get("VRT_INPUT")
if not input_file:
    raise vs.Error("VRT_INPUT environment variable not set. Use run_motion_deblur.py to run this script.")

upscale_mode = os.environ.get("VRT_UPSCALE", "none")  # none | esrgan
esrgan_model = os.environ.get("VRT_ESRGAN_MODEL", "")

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

# ---------------------------------------------------------------------------
# Optional ESRGAN upscaling (after deblur)
# ---------------------------------------------------------------------------
if upscale_mode == "esrgan":
    # Real-ESRGAN super-resolution via custom .pth weights
    # Uses chunk-based processing with caching to avoid VS deadlocks.
    import torch
    import numpy as np

    if not esrgan_model or not os.path.isfile(esrgan_model):
        raise vs.Error(f"VRT_ESRGAN_MODEL not set or file not found: {esrgan_model}")

    def load_esrgan_model(model_path, device):
        """Load a Real-ESRGAN .pth state dict and infer architecture."""
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        from models.network_rrdbnet import RRDBNet

        body_keys = [k for k in state_dict if k.startswith("body.") and ".rdb" in k]
        if body_keys:
            block_nums = set(int(k.split(".")[1]) for k in body_keys)
            num_block = max(block_nums) + 1
        else:
            num_block = 23

        conv_first_ch = state_dict["conv_first.weight"].shape[1]
        out_nc = state_dict["conv_last.weight"].shape[0]
        num_feat = state_dict["conv_first.weight"].shape[0]

        if conv_first_ch == out_nc * 4:
            scale = 2
            in_nc = out_nc
        elif conv_first_ch == out_nc * 16:
            scale = 1
            in_nc = out_nc
        else:
            scale = 4
            in_nc = conv_first_ch

        model = RRDBNet(in_nc, out_nc, num_feat=num_feat,
                        num_block=num_block, num_grow_ch=32,
                        scale=scale)
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(device)
        return model, scale, out_nc

    # Result is still RGBS from deblur, use directly
    clip_rgb = result

    # Load model eagerly
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model, sr_scale, _out_nc = load_esrgan_model(esrgan_model, _device)
    _num_input_frames = _out_nc // 3

    out_w = clip_rgb.width * sr_scale
    out_h = clip_rgb.height * sr_scale
    total_frames = clip_rgb.num_frames

    # Chunk-based caching (like vsrvrt preview_mode)
    _chunk_size = 64
    _chunk_cache = {}
    _max_cached_chunks = 3

    def _process_chunk(chunk_idx):
        """Process all frames in a chunk and cache results."""
        start = chunk_idx * _chunk_size
        end = min(start + _chunk_size, total_frames)
        nf = _num_input_frames
        half = nf // 2 if nf > 1 else 0

        results = []
        for n in range(start, end):
            if nf > 1:
                frames_rgb = []
                for offset in range(-half, -half + nf):
                    idx = min(max(n + offset, 0), total_frames - 1)
                    fr = clip_rgb.get_frame(idx)
                    planes = [np.asarray(fr[p]).copy() for p in range(3)]
                    frames_rgb.append(np.stack(planes, axis=0))
                img = np.concatenate(frames_rgb, axis=0)
            else:
                fr = clip_rgb.get_frame(n)
                planes = [np.asarray(fr[p]).copy() for p in range(3)]
                img = np.stack(planes, axis=0)

            tensor = torch.from_numpy(img).unsqueeze(0).to(_device)
            with torch.no_grad():
                output = _model(tensor)
            output_np = output.squeeze(0).clamp(0, 1).cpu().numpy()
            results.append(output_np)
            del tensor

        torch.cuda.empty_cache()
        return results

    blank = core.std.BlankClip(clip_rgb, width=out_w, height=out_h)

    def esrgan_process(n, f):
        """Serve frame from chunk cache; process chunk if not cached."""
        chunk_idx = n // _chunk_size

        if chunk_idx not in _chunk_cache:
            _chunk_cache[chunk_idx] = _process_chunk(chunk_idx)
            if len(_chunk_cache) > _max_cached_chunks:
                oldest = min(k for k in _chunk_cache if k != chunk_idx)
                del _chunk_cache[oldest]

        frame_in_chunk = n - (chunk_idx * _chunk_size)
        output_np = _chunk_cache[chunk_idx][frame_in_chunk]

        out_frame = f.copy()
        for p in range(3):
            np.copyto(np.asarray(out_frame[p]), output_np[p])
        return out_frame

    result = core.std.ModifyFrame(blank, clips=[blank], selector=esrgan_process)

# ---------------------------------------------------------------------------
# DAR correction — resize to square pixels (--dar correct)
# ---------------------------------------------------------------------------
if os.environ.get("VRT_DAR_CORRECT") == "1":
    sar_num = int(os.environ.get("VRT_SAR_NUM", "1"))
    sar_den = int(os.environ.get("VRT_SAR_DEN", "1"))
    if sar_num != sar_den:
        sar = sar_num / sar_den
        if sar > 1:
            # Wide pixels: shrink height to match
            new_h = int(round(result.height / sar / 2)) * 2
            result = core.resize.Lanczos(result, width=result.width, height=new_h)
        else:
            # Tall pixels: shrink width to match
            new_w = int(round(result.width * sar / 2)) * 2
            result = core.resize.Lanczos(result, width=new_w, height=result.height)

# Convert back to YUV for y4m output (444 to match FFV1 lossless output)
result = core.resize.Bicubic(result, format=vs.YUV444P10, matrix_s="470bg")

result.set_output()

# To disable the motion mask and apply deblurring uniformly to all areas,
# replace the "result" lines above with:
#   result = core.resize.Bicubic(deblurred, format=vs.YUV444P10, matrix_s="470bg")
#   result.set_output()
# The motion mask code (lines 26-58) can be left in place — it will be
# optimized away by VapourSynth since nothing references its output.
