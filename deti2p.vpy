import vapoursynth as vs
import os
import json
import sys

core = vs.core
core.max_cache_size = 4096  # 4GB — field analysis can be memory-hungry

# ---------------------------------------------------------------------------
# Environment variables (set by run_deti2p.py)
# ---------------------------------------------------------------------------
input_file = os.environ.get("VRT_INPUT")
if not input_file:
    raise vs.Error("VRT_INPUT environment variable not set. Use run_deti2p.py to run this script.")

mode = os.environ.get("VRT_MODE", "auto")          # auto | ivtc | deinterlace
field_order = os.environ.get("VRT_FIELD_ORDER", "auto")  # auto | tff | bff
upscale_mode = os.environ.get("VRT_UPSCALE", "none")     # none | vsr | esrgan
esrgan_model = os.environ.get("VRT_ESRGAN_MODEL", "")    # path to .pth
upscale_width = int(os.environ.get("VRT_WIDTH", "0"))
upscale_height = int(os.environ.get("VRT_HEIGHT", "0"))

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

# ---------------------------------------------------------------------------
# 1. Source
# ---------------------------------------------------------------------------
clip = core.ffms2.Source(input_file)
_matrix = get_matrix(clip)  # Detect matrix from source before any processing

src_fps_num = clip.fps.numerator
src_fps_den = clip.fps.denominator
src_fps = src_fps_num / src_fps_den

# ---------------------------------------------------------------------------
# 2. Auto-detect field order (TFF vs BFF)
# ---------------------------------------------------------------------------
def detect_field_order(clip):
    """Detect field order by analyzing _FieldBased frame property.

    ffms2 sets _FieldBased: 0=progressive, 1=BFF, 2=TFF.
    Sample several frames and take the majority vote.
    """
    sample_count = min(100, clip.num_frames)
    tff_count = 0
    bff_count = 0
    prog_count = 0

    for i in range(0, clip.num_frames, max(1, clip.num_frames // sample_count)):
        try:
            frame = clip.get_frame(i)
        except vs.Error:
            continue
        fb = frame.props.get("_FieldBased", 0)
        if fb == 2:
            tff_count += 1
        elif fb == 1:
            bff_count += 1
        else:
            prog_count += 1

    total = tff_count + bff_count + prog_count
    if total == 0:
        return "tff", "unknown"

    # If mostly progressive, no deinterlacing needed
    if prog_count > total * 0.8:
        return "tff", "progressive"

    if tff_count >= bff_count:
        return "tff", "interlaced"
    else:
        return "bff", "interlaced"

# ---------------------------------------------------------------------------
# 3. Auto-detect telecine vs interlace
# ---------------------------------------------------------------------------
def detect_telecine(clip, order):
    """Analyze a sample of frames with VFM to detect telecine pattern.

    Telecined content (3:2 pulldown) will show ~20% combed frames after
    field matching. True interlaced content shows much higher combed rates.

    Returns: "telecine" | "interlaced" | "progressive"
    """
    ord_val = 1 if order == "tff" else 0

    # Run VFM on a sample to check combed frame ratio
    sample_frames = min(500, clip.num_frames)
    sample = core.std.Trim(clip, 0, sample_frames - 1)
    matched = core.vivtc.VFM(sample, order=ord_val, mode=5)

    combed_count = 0
    matched_count = 0
    for i in range(matched.num_frames):
        try:
            frame = matched.get_frame(i)
        except vs.Error:
            continue
        combed = frame.props.get("_Combed", 0)
        if combed:
            combed_count += 1
        matched_count += 1

    if matched_count == 0:
        return "interlaced"

    combed_ratio = combed_count / matched_count

    # Telecined content: VFM matches most fields cleanly, few combed remain
    # After VDecimate ~20% of frames get dropped (the duplicate pulldown fields)
    if combed_ratio < 0.15:
        return "telecine"
    elif combed_ratio < 0.40:
        # Hybrid — some telecine, some interlace (common with mixed content)
        return "telecine"
    else:
        return "interlaced"


# ---------------------------------------------------------------------------
# 4. Determine processing mode
# ---------------------------------------------------------------------------
if field_order == "auto":
    detected_order, field_type = detect_field_order(clip)
else:
    detected_order = field_order
    field_type = "interlaced"  # user overrode, assume interlaced

ord_val = 1 if detected_order == "tff" else 0

if mode == "auto":
    if field_type == "progressive":
        detected_mode = "progressive"
    else:
        detected_mode = detect_telecine(clip, detected_order)
elif mode == "ivtc":
    detected_mode = "telecine"
elif mode == "deinterlace":
    detected_mode = "interlaced"
elif mode == "progressive":
    detected_mode = "progressive"
else:
    detected_mode = "telecine"

# Write detection results for the runner to display
analysis = {
    "field_order": detected_order,
    "field_type": field_type,
    "mode": detected_mode,
}
analysis_path = os.environ.get("VRT_ANALYSIS_OUT", "")
if analysis_path:
    with open(analysis_path, "w") as f:
        json.dump(analysis, f)

# If analysis-only mode, output a single frame and exit
if os.environ.get("VRT_ANALYZE_ONLY") == "1":
    core.std.Trim(clip, 0, 0).set_output()

else:
    # -----------------------------------------------------------------------
    # 5. Process: IVTC or Deinterlace
    # -----------------------------------------------------------------------
    if detected_mode == "telecine":
        # Hard telecine: VFM field matching → VDecimate to drop duplicates
        # order: 0=BFF, 1=TFF
        # mode=5: highest quality matching (2-way + 3rd match)
        clip = core.vivtc.VFM(clip, order=ord_val, mode=5)
        clip = core.vivtc.VDecimate(clip)
        # Output: 23.976 fps progressive from 29.97 fps telecined source

    elif detected_mode == "interlaced":
        # True deinterlace using nnedi3 (bob to double rate, then decimate)
        # field: 0=same rate(keep bottom), 1=same rate(keep top),
        #        2=double rate(start bottom), 3=double rate(start top)
        # For same-rate deinterlace: use field=1 for TFF, field=0 for BFF
        nnedi3_field = 1 if detected_order == "tff" else 0
        clip = core.nnedi3.nnedi3(clip, field=nnedi3_field, qual=2, nsize=0, nns=3)

    # else: progressive — no processing needed

    # -------------------------------------------------------------------
    # 6. Optional upscale
    # -------------------------------------------------------------------
    if upscale_mode == "vsr":
        # VapourSynth RVRT video super-resolution (4x)
        from vsrvrt import SuperRes
        # Convert to RGBS for RVRT
        clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s=_matrix)
        clip = SuperRes(clip, model="reds", preview_mode=True,
                        chunk_overlap=24)
        # Back to YUV for output
        clip = core.resize.Bicubic(clip, format=vs.YUV444P10, matrix_s=_matrix)

    elif upscale_mode == "esrgan":
        # Real-ESRGAN single-image super-resolution via custom .pth weights
        # Uses chunk-based processing with caching to avoid VS deadlocks.
        # (Same approach as vsrvrt's preview_mode)
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

        # Convert to RGBS for ESRGAN
        clip_rgb = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s=_matrix)

        # Load model eagerly
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model, sr_scale, _out_nc = load_esrgan_model(esrgan_model, _device)
        _num_input_frames = _out_nc // 3

        out_w = clip_rgb.width * sr_scale
        out_h = clip_rgb.height * sr_scale
        total_frames = clip_rgb.num_frames

        # Chunk-based caching (like vsrvrt preview_mode)
        # Process frames in chunks; cache results to avoid repeated get_frame calls
        _chunk_size = 64
        _chunk_cache = {}  # chunk_idx -> list of numpy arrays
        _max_cached_chunks = 3  # Keep only recent chunks in memory

        def _process_chunk(chunk_idx):
            """Process all frames in a chunk and cache results."""
            start = chunk_idx * _chunk_size
            end = min(start + _chunk_size, total_frames)
            nf = _num_input_frames
            half = nf // 2 if nf > 1 else 0

            results = []
            for n in range(start, end):
                # Build input for this frame
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

                # Run inference
                tensor = torch.from_numpy(img).unsqueeze(0).to(_device)
                with torch.no_grad():
                    output = _model(tensor)
                output_np = output.squeeze(0).clamp(0, 1).cpu().numpy()
                results.append(output_np)
                del tensor

            torch.cuda.empty_cache()
            return results

        # Create blank output template
        blank = core.std.BlankClip(clip_rgb, width=out_w, height=out_h)

        def esrgan_process(n, f):
            """Serve frame from chunk cache; process chunk if not cached."""
            chunk_idx = n // _chunk_size

            # Process chunk if not in cache
            if chunk_idx not in _chunk_cache:
                _chunk_cache[chunk_idx] = _process_chunk(chunk_idx)

                # Evict old chunks to bound memory
                if len(_chunk_cache) > _max_cached_chunks:
                    oldest = min(k for k in _chunk_cache if k != chunk_idx)
                    del _chunk_cache[oldest]

            # Get frame from cache
            frame_in_chunk = n - (chunk_idx * _chunk_size)
            output_np = _chunk_cache[chunk_idx][frame_in_chunk]

            # Build output frame - f is the blank frame passed by VS
            out_frame = f.copy()
            for p in range(3):
                np.copyto(np.asarray(out_frame[p]), output_np[p])
            return out_frame

        clip = core.std.ModifyFrame(blank, clips=[blank], selector=esrgan_process)

        # Back to YUV
        clip = core.resize.Bicubic(clip, format=vs.YUV444P10, matrix_s=_matrix)

    else:
        # No upscale — just ensure output is YUV444P10
        if clip.format.id != vs.YUV444P10:
            if clip.format.color_family == vs.RGB:
                clip = core.resize.Bicubic(clip, format=vs.YUV444P10, matrix_s=_matrix)
            else:
                clip = core.resize.Bicubic(clip, format=vs.YUV444P10)

    # -------------------------------------------------------------------
    # 7. DAR correction — resize to square pixels (--dar correct)
    #    Prefers shrinking the larger axis via Lanczos for sharpness.
    # -------------------------------------------------------------------
    if os.environ.get("VRT_DAR_CORRECT") == "1":
        sar_num = int(os.environ.get("VRT_SAR_NUM", "1"))
        sar_den = int(os.environ.get("VRT_SAR_DEN", "1"))
        if sar_num != sar_den:
            sar = sar_num / sar_den
            if sar > 1:
                # Wide pixels: shrink height to match
                new_h = int(round(clip.height / sar / 2)) * 2
                clip = core.resize.Lanczos(clip, width=clip.width, height=new_h)
            else:
                # Tall pixels: shrink width to match
                new_w = int(round(clip.width * sar / 2)) * 2
                clip = core.resize.Lanczos(clip, width=new_w, height=clip.height)

    # -------------------------------------------------------------------
    # 8. Optional resize (after upscale / DAR correction)
    #    Specifying only --width or --height keeps the other dimension as-is.
    # -------------------------------------------------------------------
    rw = upscale_width if upscale_width > 0 else clip.width
    rh = upscale_height if upscale_height > 0 else clip.height
    if rw != clip.width or rh != clip.height:
        clip = core.resize.Spline36(clip, width=rw, height=rh)

    clip.set_output()
