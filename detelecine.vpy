import vapoursynth as vs
import os
import json
import sys

core = vs.core
core.max_cache_size = 4096  # 4GB — field analysis can be memory-hungry

# ---------------------------------------------------------------------------
# Environment variables (set by run_detelecine.py)
# ---------------------------------------------------------------------------
input_file = os.environ.get("VRT_INPUT")
if not input_file:
    raise vs.Error("VRT_INPUT environment variable not set. Use run_detelecine.py to run this script.")

mode = os.environ.get("VRT_MODE", "auto")          # auto | ivtc | deinterlace
field_order = os.environ.get("VRT_FIELD_ORDER", "auto")  # auto | tff | bff
upscale_mode = os.environ.get("VRT_UPSCALE", "none")     # none | vsr | esrgan
esrgan_model = os.environ.get("VRT_ESRGAN_MODEL", "")    # path to .pth
upscale_width = int(os.environ.get("VRT_WIDTH", "0"))
upscale_height = int(os.environ.get("VRT_HEIGHT", "0"))

# ---------------------------------------------------------------------------
# 1. Source
# ---------------------------------------------------------------------------
clip = core.ffms2.Source(input_file)
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
        frame = clip.get_frame(i)
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
        frame = matched.get_frame(i)
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
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# 5. Process: IVTC or Deinterlace
# ---------------------------------------------------------------------------
if detected_mode == "telecine":
    # Inverse telecine: VFM field matching → VDecimate to drop duplicates
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

# ---------------------------------------------------------------------------
# 6. Optional upscale
# ---------------------------------------------------------------------------
if upscale_mode == "vsr":
    # VapourSynth RVRT video super-resolution (4x)
    from vsrvrt import SuperRes
    # Convert to RGBS for RVRT
    clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="470bg")
    clip = SuperRes(clip, model="reds", preview_mode=True,
                    tile_size=(16, 0, 0), tile_overlap=(4, 0, 0),
                    chunk_overlap=24)
    # Back to YUV for output
    clip = core.resize.Bicubic(clip, format=vs.YUV444P10, matrix_s="470bg")

elif upscale_mode == "esrgan":
    # Real-ESRGAN single-image super-resolution via custom .pth weights
    # This uses torch directly — load model, process frame-by-frame
    import torch
    import numpy as np

    if not esrgan_model or not os.path.isfile(esrgan_model):
        raise vs.Error(f"VRT_ESRGAN_MODEL not set or file not found: {esrgan_model}")

    # Lazy-load the model on first frame
    _esrgan_net = [None]
    _esrgan_scale = [4]  # will be detected from model

    def load_esrgan_model(model_path, device):
        """Load a Real-ESRGAN .pth state dict and infer architecture."""
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        # Detect RRDBNet architecture from state dict keys
        # Standard RRDBNet: conv_first, body.{0..N}.rdb{1,2,3}, conv_body,
        #   conv_up1, conv_up2, [conv_up3], conv_hr, conv_last
        from models.network_rrdbnet import RRDBNet

        # Infer num_block from body keys
        body_keys = [k for k in state_dict if k.startswith("body.") and ".rdb" in k]
        if body_keys:
            block_nums = set(int(k.split(".")[1]) for k in body_keys)
            num_block = max(block_nums) + 1
        else:
            num_block = 23  # default

        # Infer scale from conv_up keys
        up_keys = [k for k in state_dict if k.startswith("conv_up")]
        scale = 2 ** len(up_keys) if up_keys else 4

        # Infer in/out channels from first/last conv
        in_nc = state_dict["conv_first.weight"].shape[1]
        out_nc = state_dict["conv_last.weight"].shape[0]
        num_feat = state_dict["conv_first.weight"].shape[0]

        model = RRDBNet(in_nc, out_nc, num_feat=num_feat,
                        num_block=num_block, num_grow_ch=32,
                        scale=scale)
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(device)
        return model, scale

    def esrgan_process(n, f):
        """Frame-by-frame ESRGAN processing callback."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if _esrgan_net[0] is None:
            _esrgan_net[0], _esrgan_scale[0] = load_esrgan_model(esrgan_model, device)

        model = _esrgan_net[0]
        scale = _esrgan_scale[0]

        # Read RGBS frame to numpy → torch
        planes = [np.asarray(f[0][p]) for p in range(f[0].format.num_planes)]
        img = np.stack(planes, axis=0)  # [C, H, W] float32
        tensor = torch.from_numpy(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)

        output = output.squeeze(0).clamp(0, 1).cpu().numpy()
        h, w = output.shape[1], output.shape[2]

        # Create output frame
        out_frame = f[0].copy()
        for p in range(3):
            np.copyto(np.asarray(out_frame[p]), output[p])
        return out_frame

    # Convert to RGBS for ESRGAN
    clip_rgb = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="470bg")

    # We need to create the output clip with the correct dimensions
    # Use a blank clip at the target size as template
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_state = torch.load(esrgan_model, map_location="cpu", weights_only=False)
    if "params_ema" in test_state:
        test_state = test_state["params_ema"]
    elif "params" in test_state:
        test_state = test_state["params"]
    up_keys = [k for k in test_state if k.startswith("conv_up")]
    sr_scale = 2 ** len(up_keys) if up_keys else 4
    del test_state

    out_w = clip_rgb.width * sr_scale
    out_h = clip_rgb.height * sr_scale

    blank = core.std.BlankClip(clip_rgb, width=out_w, height=out_h)
    clip = core.std.ModifyFrame(blank, clips=[clip_rgb], selector=esrgan_process)

    # Back to YUV
    clip = core.resize.Bicubic(clip, format=vs.YUV444P10, matrix_s="470bg")

else:
    # No upscale — just ensure output is YUV444P10
    if clip.format.id != vs.YUV444P10:
        matrix = "470bg"  # BT.601 for SD content
        if clip.width >= 1280 or clip.height >= 720:
            matrix = "709"
        if clip.format.color_family == vs.RGB:
            clip = core.resize.Bicubic(clip, format=vs.YUV444P10, matrix_s=matrix)
        else:
            clip = core.resize.Bicubic(clip, format=vs.YUV444P10)

# ---------------------------------------------------------------------------
# 7. Optional resize (after upscale, or standalone)
# ---------------------------------------------------------------------------
if upscale_width > 0 and upscale_height > 0:
    clip = core.resize.Spline36(clip, width=upscale_width, height=upscale_height)

clip.set_output()
