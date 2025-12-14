#!/usr/bin/env python3
import os
import sys
import types

import torch
try:
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None
except Exception:
    pass

import torchaudio
import soundfile as sf
import numpy as np
import inspect

# --------- Basic info (handy for debugging once) ----------
print("torch version:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torchaudio version:", torchaudio.__version__)

# =========================================================
# 1) Stub out voxcpm.zipenhancer so ModelScope never runs
# =========================================================
zipenhancer_stub = types.ModuleType("voxcpm.zipenhancer")
zipenhancer_stub.__package__ = "voxcpm"


class ZipEnhancer:
    """No-op denoiser stub – VoxCPM will behave as if denoiser is disabled."""

    def __init__(self, *args, **kwargs):
        print("[Stub ZipEnhancer] Initialized (denoiser disabled).")

    def __call__(self, wav, sr):
        # passthrough
        return wav, sr


zipenhancer_stub.ZipEnhancer = ZipEnhancer
sys.modules["voxcpm.zipenhancer"] = zipenhancer_stub

# =========================================================
# 2) Patch torchaudio.load -> use soundfile + float32
# =========================================================
try:
    import torchaudio._torchcodec as _tc  # type: ignore[attr-defined]

    print("Found torchaudio._torchcodec, patching load_with_torchcodec -> soundfile...")

    def _sf_load_fallback(path, *args, **kwargs):
        """Fallback loader using soundfile, returning float32 tensor."""
        audio, sr = sf.read(path, always_2d=False)
        # (time,) or (time, channels)
        if audio.ndim == 1:
            tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, time)
        else:
            tensor = torch.from_numpy(audio.T)  # (time, ch) -> (ch, time)
        tensor = tensor.to(torch.float32)
        return tensor, sr

    # Replace internal torchcodec loader
    _tc.load_with_torchcodec = _sf_load_fallback  # type: ignore[attr-defined]

    # Make torchaudio.load call our fallback too
    def _safe_torchaudio_load(path, *args, **kwargs):
        return _sf_load_fallback(path, *args, **kwargs)

    torchaudio.load = _safe_torchaudio_load  # type: ignore[assignment]

except ImportError:
    print("torchaudio._torchcodec not found; leaving torchaudio.load unchanged.")

# =========================================================
# 3) Import VoxCPM (now safe)
# =========================================================
from voxcpm import VoxCPM  # noqa: E402

# =========================================================
# 4) Load model once, keep it in memory
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

print("Loading VoxCPM1.5 (with ZipEnhancer stub)...")
VOXCPM_MODEL = VoxCPM.from_pretrained("openbmb/VoxCPM1.5")
SAMPLE_RATE = getattr(VOXCPM_MODEL.tts_model, "sample_rate", 44100)
print(f"VoxCPM sample rate: {SAMPLE_RATE} Hz")

# Limit how much of the Marcus transcript we feed as prompt text
MAX_VOXCPM_PROMPT_CHARS = 640

# =========================================================
# Default Marcus transcript matching marcus_ref.wav
# (ends at "nothing more.")
# =========================================================
DEFAULT_MARCUS_PROMPT = (
    "Driving my truck home on a cold February night in West Texas, after eighteen hours "
    "on a construction site, the last thing I wanted was a flat tire. The highway was a "
    "two lane rumor, the sky inked in hard. When the tire thumped and dragged, I eased "
    "onto the shoulder, hazards ticking at the mesquite. I left the engine idling for "
    "heat and told myself this was ten minutes of cold and cussing, nothing more."
)

def _truncate_prompt_text(prompt_text: str, limit: int = MAX_VOXCPM_PROMPT_CHARS) -> str:
    """Trim prompt text to a char limit, but cut on a word boundary if possible."""
    prompt_text = prompt_text.strip()
    if len(prompt_text) <= limit:
        return prompt_text

    cut = prompt_text.rfind(" ", 0, limit)
    if cut == -1:
        cut = limit
    return prompt_text[:cut].strip()


def _quality_to_params(quality: str):
    """
    Map human-friendly quality setting to cfg_value and timesteps.

    - fast  : quickest, less expressive
    - vivid : nice balance of expressiveness & speed
    - max   : most detailed / slowest
    """
    q = (quality or "vivid").lower()

    if q == "fast":
        return 1.8, 10
    elif q == "max":
        return 2.5, 24
    else:  # vivid (default)
        return 2.0, 20


def _trim_leading_trailing_silence(
    wav: np.ndarray,
    sr: int,
    *,
    rel_thresh: float = 0.015,    # 1% of peak (same as you had)
    abs_floor: float = 1e-3,      # don't go below this
    keep_head_s: float = 0.1,     # keep 100ms before speech
    keep_tail_s: float = 0.5,     # keep 500ms after speech
    min_seg_s: float = 0.25,      # ignore very tiny "speech" islands (<250ms)
    drop_tail_if_after_gap_s: float = 0.20,  # if a tiny tail happens after a big silence, drop it
):
    if wav.size == 0:
        return wav

    peak = float(np.max(np.abs(wav)))
    if peak <= 0:
        return wav

    thresh = max(abs_floor, rel_thresh * peak)
    mask = np.abs(wav) > thresh
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return wav

    # Build contiguous segments from idx
    breaks = np.where(np.diff(idx) > 1)[0]
    seg_starts = np.r_[idx[0], idx[breaks + 1]]
    seg_ends   = np.r_[idx[breaks], idx[-1]]

    # Filter out tiny segments
    seg_durs = (seg_ends - seg_starts + 1) / sr
    keep = seg_durs >= min_seg_s

    # If everything is tiny, fall back to original first/last
    if not np.any(keep):
        first_idx = int(idx[0])
        last_idx = int(idx[-1])
    else:
        ks = seg_starts[keep]
        ke = seg_ends[keep]

        # Special handling: if the LAST kept segment is tiny and comes after a big gap, drop it
        # (this is what usually kills the "random last syllable" problem)
        if len(ks) >= 2:
            last_seg_len_s = (ke[-1] - ks[-1] + 1) / sr
            gap_before_last_s = (ks[-1] - ke[-2]) / sr
            if last_seg_len_s < (min_seg_s * 1.2) and gap_before_last_s > drop_tail_if_after_gap_s:
                ks = ks[:-1]
                ke = ke[:-1]

        first_idx = int(ks[0])
        last_idx = int(ke[-1])

    head = max(0, first_idx - int(keep_head_s * sr))
    tail = min(len(wav), last_idx + int(keep_tail_s * sr))

    return wav[head:tail]


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "")
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "")
    if not v:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off")


def _cap_internal_silences(
    wav: np.ndarray,
    sr: int,
    *,
    rel_thresh: float = 0.006,
    abs_floor: float = 3e-4,
    smooth_win_s: float = 0.02,
    min_silence_s: float = 0.30,
    max_silence_s: float = 0.55,
    target_silence_s: float = 0.18,
    edge_guard_s: float = 0.05,
) -> np.ndarray:
    """
    Reduce overly-long *internal* silences (between sentences) inside an already-trimmed wav.

    Strategy:
      - compute a smoothed amplitude envelope
      - find silent segments
      - for any silent segment longer than `max_silence_s`, cut the *middle* so remaining
        silence is ~`target_silence_s` (keeping silence at both edges to avoid clicks)
    """
    if wav.size == 0:
        return wav

    peak = float(np.max(np.abs(wav)))
    if peak <= 0:
        return wav

    thr = max(abs_floor, rel_thresh * peak)

    win = max(1, int(sr * smooth_win_s))
    kernel = (np.ones(win, dtype=np.float32) / float(win))
    env = np.convolve(np.abs(wav).astype(np.float32), kernel, mode="same")

    silent = env < thr
    if not np.any(silent):
        return wav

    changes = np.diff(silent.astype(np.int8))
    starts = np.where(changes == 1)[0] + 1   # False->True
    ends   = np.where(changes == -1)[0] + 1  # True->False

    if silent[0]:
        starts = np.r_[0, starts]
    if silent[-1]:
        ends = np.r_[ends, len(wav)]

    if starts.size == 0 or ends.size == 0:
        return wav

    guard = int(edge_guard_s * sr)
    min_len = int(min_silence_s * sr)
    max_len = int(max_silence_s * sr)
    target_len = int(target_silence_s * sr)

    if target_len < 0:
        target_len = 0

    out_chunks = []
    cursor = 0
    removed = 0
    capped = 0

    for s, e in zip(starts, ends):
        # only internal segments
        if s <= guard or e >= (len(wav) - guard):
            continue

        seg_len = e - s
        if seg_len < min_len:
            continue
        if seg_len <= max_len:
            continue
        if target_len >= seg_len:
            continue

        # keep some silence at both sides; remove from the middle
        left_keep = target_len // 2
        right_keep = target_len - left_keep

        cut_start = s + left_keep
        cut_end = e - right_keep

        if cut_end <= cut_start:
            continue

        # append audio up to the cut, skip the middle, continue after cut
        out_chunks.append(wav[cursor:cut_start])
        cursor = cut_end

        removed += (cut_end - cut_start)
        capped += 1

    if capped == 0:
        return wav

    out_chunks.append(wav[cursor:])
    out = np.concatenate(out_chunks, axis=0)

    if removed > 0:
        print(
            f"[Marcus TTS] Capped internal pauses: capped={capped}, "
            f"removed={(removed/sr):.2f}s"
        )

    return out


def _generate_with_early_stop(
    *,
    text: str,
    prompt_wav_path: str | None,
    prompt_text: str | None,
    cfg_value: float,
    inference_timesteps: int,
    badcase_ratio_threshold: float,
    trailing_silence_stop_s: float = 6.0,
):
    # Fallback to non-streaming if streaming isn't available
    if not hasattr(VOXCPM_MODEL, "generate_streaming") or trailing_silence_stop_s <= 0:
        return VOXCPM_MODEL.generate(
            text=text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=False,
            denoise=False,
            retry_badcase=True,
            retry_badcase_max_times=2,
            retry_badcase_ratio_threshold=badcase_ratio_threshold,
        ).astype(np.float32)

    gen_fn = VOXCPM_MODEL.generate_streaming
    sig = inspect.signature(gen_fn)

    kwargs = dict(
        text=text,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        normalize=False,
        denoise=False,
        retry_badcase=True,
        retry_badcase_max_times=2,
        retry_badcase_ratio_threshold=badcase_ratio_threshold,
    )
    # Only pass args the installed VoxCPM actually accepts
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    silence_samples = int(trailing_silence_stop_s * SAMPLE_RATE)
    collected = []
    running_peak = 0.0
    last_non_silent_global = 0
    total = 0

    for chunk in gen_fn(**kwargs):
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.size == 0:
            continue

        collected.append(chunk)

        # Update running peak + dynamic threshold (same idea as your final trim)
        peak = float(np.max(np.abs(chunk)))
        running_peak = max(running_peak, peak)
        thresh = max(1e-3, 0.01 * running_peak) if running_peak > 0 else 1e-3

        idx = np.where(np.abs(chunk) > thresh)[0]
        if idx.size > 0:
            last_non_silent_global = total + int(idx[-1])

        total += len(chunk)

        # If we've seen trailing silence long enough, stop pulling more audio
        if total - last_non_silent_global > silence_samples:
            print(f"[Marcus TTS] Early-stopping streaming after ~{(total/SAMPLE_RATE):.2f}s (trailing silence).")
            break

    if not collected:
        return np.zeros((0,), dtype=np.float32)

    return np.concatenate(collected, axis=0)


# =========================================================
# 5) Public helper: synthesize_marcus(...)
# =========================================================
def synthesize_marcus(
    text: str,
    out_path: str = "marcus_tts.wav",
    ref_wav: str = "ref/marcus_ref.wav",
    prompt_text: str | None = None,
    *,
    use_ref_audio: bool = True,
    quality: str = "vivid",
    cfg_value: float | None = None,
    inference_timesteps: int | None = None,
    badcase_ratio_threshold: float = 6.0,
) -> str:
    """
    Generate speech in Marcus's cloned voice using VoxCPM1.5.

    Parameters
    ----------
    text : str
        What Marcus should say.
    out_path : str
        Output WAV file path.
    ref_wav : str
        Reference Marcus voice clip.
    prompt_text : str | None
        Transcript / style text corresponding to `ref_wav`. If None, uses DEFAULT_MARCUS_PROMPT.
    use_ref_audio : bool
        If True, use the Marcus reference audio for zero-shot voice cloning.
    quality : {"fast","vivid","max"}
        Convenience knob for cfg / timesteps.
    cfg_value : float | None
        Override guidance strength. If None, derived from `quality`.
    inference_timesteps : int | None
        Override diffusion steps. If None, derived from `quality`.
    badcase_ratio_threshold : float
        Passed through to VoxCPM. With retry_badcase=False, it won't trigger reruns.
    """

    text = (text or "").strip()
    if not text:
        raise ValueError("synthesize_marcus(): `text` is empty.")

    # Only enforce ref_wav presence if we're actually using it
    if use_ref_audio and not os.path.exists(ref_wav):
        raise FileNotFoundError(
            f"Reference file '{ref_wav}' not found in {os.getcwd()}.\n"
            "Place your Marcus reference voice clip there or pass the correct path."
        )

    # Fill default prompt text (Marcus reference transcript) if needed
    if prompt_text is None:
        prompt_text = DEFAULT_MARCUS_PROMPT

    model_prompt_text = _truncate_prompt_text(prompt_text, MAX_VOXCPM_PROMPT_CHARS)

    # Derive cfg / timesteps from quality if not explicitly provided
    auto_cfg, auto_steps = _quality_to_params(quality)
    if cfg_value is None:
        cfg_value = auto_cfg
    if inference_timesteps is None:
        inference_timesteps = auto_steps

    prompt_preview = model_prompt_text[:148].replace("\n", " ")
    if len(model_prompt_text) > 148:
        prompt_preview += " ..."

    print(f"[Marcus TTS] text to speak: {text!r}")
    print(
        f"[Marcus TTS] use_ref_audio={use_ref_audio}, "
        f"quality={quality}, cfg_value={cfg_value}, "
        f"inference_timesteps={inference_timesteps}, "
        f"badcase_ratio_threshold={badcase_ratio_threshold}"
    )

    if use_ref_audio:
        model_prompt_wav_path = ref_wav
        print(f"[Marcus TTS] model_prompt_wav_path={model_prompt_wav_path!r}")
        print(f"[Marcus TTS] model_prompt_text_preview={prompt_preview!r}")
    else:
        model_prompt_wav_path = None
        model_prompt_text = None
        print("[Marcus TTS] Zero-shot text-only mode (no reference audio).")

    # ---------------------------------------------
    # IMPORTANT: badcase handling
    # ---------------------------------------------
    wav = VOXCPM_MODEL.generate(
        text=text,
        prompt_wav_path=model_prompt_wav_path,
        prompt_text=model_prompt_text if use_ref_audio else None,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        normalize=False,
        denoise=False,             # IMPORTANT: avoid external ZipEnhancer path
        retry_badcase=True,
        retry_badcase_max_times=2, # MUST be >= 1 to avoid latent_pred bug
        retry_badcase_ratio_threshold=badcase_ratio_threshold,
    )

    wav = wav.astype(np.float32)

    # 1) Trim leading/trailing junk & long tail
    before = len(wav) / SAMPLE_RATE
    wav = _trim_leading_trailing_silence(wav, SAMPLE_RATE)
    after = len(wav) / SAMPLE_RATE
    if after < before:
        print(f"[Marcus TTS] Trim lead+trail: {before:.2f}s -> {after:.2f}s")

    # 2) NEW: Cap overly-long *internal* pauses between sentences
    cap_enabled = _env_bool("MARCUS_CAP_INTERNAL_SILENCE", True)
    if cap_enabled:
        cap_before = len(wav) / SAMPLE_RATE
        wav = _cap_internal_silences(
            wav,
            SAMPLE_RATE,
            rel_thresh=_env_float("MARCUS_INTERNAL_REL_THRESH", 0.006),
            abs_floor=_env_float("MARCUS_INTERNAL_ABS_FLOOR", 3e-4),
            smooth_win_s=_env_float("MARCUS_INTERNAL_SMOOTH_WIN_S", 0.02),
            min_silence_s=_env_float("MARCUS_INTERNAL_MIN_SILENCE_S", 0.30),
            max_silence_s=_env_float("MARCUS_INTERNAL_MAX_SILENCE_S", 0.55),
            target_silence_s=_env_float("MARCUS_INTERNAL_TARGET_SILENCE_S", 0.18),
            edge_guard_s=_env_float("MARCUS_INTERNAL_EDGE_GUARD_S", 0.05),
        )
        cap_after = len(wav) / SAMPLE_RATE
        if cap_after < cap_before:
            print(f"[Marcus TTS] Internal pause cap: {cap_before:.2f}s -> {cap_after:.2f}s")

    sf.write(out_path, wav, SAMPLE_RATE)
    print(f"[Marcus TTS] Saved: {out_path}  ({len(wav) / SAMPLE_RATE:.2f} s)")
    return out_path


# =========================================================
# 6) CLI usage
# =========================================================
if __name__ == "__main__":
    demo_text = (
        "At two thirteen A M, Marcus drove alone through the empty West Texas highway, "
        "wondering why his watch had started running backwards."
    )
    synthesize_marcus(
        demo_text,
        out_path="marcus_demo_voxcpm.wav",
        quality="vivid",
    )
