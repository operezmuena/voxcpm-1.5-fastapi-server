#!/usr/bin/env python3
import os
import re
import uuid
from typing import List, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

# Heavy import (loads VoxCPM model). Keep after FastAPI is defined? It's fine either way,
# but having FastAPI defined early makes debugging clearer.
from marcus_voxcpm import synthesize_marcus, SAMPLE_RATE

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
EPISODE_OUT_DIR = os.getenv("MARCUS_EPISODE_DIR", "/workspace/episodes")
MAX_CHARS_PER_CHUNK = int(os.getenv("MARCUS_MAX_CHARS", "512"))

# Pause defaults
DEFAULT_PAUSE_S = float(os.getenv("MARCUS_DEFAULT_PAUSE_S", "0.6"))  # <pause> uses this
AUTO_PARA_PAUSE_S = float(os.getenv("MARCUS_AUTO_PARA_PAUSE_S", "0.0"))  # 0 disables auto pause between paragraphs

os.makedirs(EPISODE_OUT_DIR, exist_ok=True)

app = FastAPI(
    title="Marcus VoxCPM TTS API",
    description="Generate Marcus-voiced episodes from text.",
    version="1.0.1",
)

# ---------------------------------------------------------
# Pause tag parsing
# ---------------------------------------------------------
# Supports:
#   <pause>
#   <pause=0.6>
#   <pause=0.6s>
#   <pause=500ms>
PAUSE_TAG_RE = re.compile(
    r"^\s*<pause(?:\s*=\s*(\d+(?:\.\d+)?)(ms|s)?)?\s*/?>\s*$",
    re.IGNORECASE,
)

INLINE_PAUSE_RE = re.compile(
    r"\s*(<pause(?:\s*=\s*\d+(?:\.\d+)?(?:ms|s)?)?\s*/?>)\s*",
    re.IGNORECASE,
)

def is_pause_tag(s: str) -> bool:
    return PAUSE_TAG_RE.match(s or "") is not None

def pause_seconds(tag: str, default_s: float = DEFAULT_PAUSE_S) -> float:
    m = PAUSE_TAG_RE.match(tag or "")
    if not m:
        return 0.0
    val, unit = m.group(1), m.group(2)
    if val is None:
        secs = default_s
    else:
        x = float(val)
        secs = x / 1000.0 if unit and unit.lower() == "ms" else x
    return max(0.0, min(10.0, secs))

# ---------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def split_paragraph(para: str, max_chars: int) -> List[str]:
    """
    Split one paragraph into chunks <= max_chars, preferring sentence boundaries.
    Falls back to word-splitting for very long sentences.
    """
    para = para.strip()
    if not para:
        return []

    if len(para) <= max_chars:
        return [para]

    sentences = _SENT_SPLIT_RE.split(para)
    chunks: List[str] = []
    cur = ""

    def flush():
        nonlocal cur
        if cur.strip():
            chunks.append(cur.strip())
        cur = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        # If sentence itself is too long, split by words
        if len(s) > max_chars:
            flush()
            words = s.split()
            wcur = ""
            for w in words:
                if not wcur:
                    wcur = w
                elif len(wcur) + 1 + len(w) <= max_chars:
                    wcur += " " + w
                else:
                    chunks.append(wcur)
                    wcur = w
            if wcur:
                chunks.append(wcur)
            continue

        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars:
            cur += " " + s
        else:
            flush()
            cur = s

    flush()
    return chunks

def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """
    Splits full episode text into chunks. Preserves <pause> tags as standalone chunks.
    Paragraphs are separated by blank lines.
    """
    text = text.replace("\r\n", "\n")

    # Make inline pause tags become their own paragraph
    text = INLINE_PAUSE_RE.sub(r"\n\n\1\n\n", text)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[str] = []
    for i, para in enumerate(paragraphs):
        if is_pause_tag(para):
            chunks.append(para.strip())
            continue

        chunks.extend(split_paragraph(para, max_chars=max_chars))

        # Optional: auto-insert a paragraph pause unless next paragraph is explicitly a pause tag
        if AUTO_PARA_PAUSE_S > 0 and i < len(paragraphs) - 1:
            nxt = paragraphs[i + 1]
            if not is_pause_tag(nxt):
                chunks.append(f"<pause={AUTO_PARA_PAUSE_S}s>")

    return chunks

def _preview(chunk: str) -> str:
    c = chunk.replace("\n", " ")
    if len(c) <= 110:
        return c
    return f"{c[:60]} ... {c[-40:]}"

# ---------------------------------------------------------
# Episode generation
# ---------------------------------------------------------
def build_episode_from_text(
    script_text: str,
    episode_id: Optional[str] = None,
    *,
    quality: str = "vivid",
    use_ref_audio: bool = True,
) -> str:
    if not episode_id:
        episode_id = f"marcus_ep_{uuid.uuid4().hex[:6]}"

    chunks = chunk_text(script_text, max_chars=MAX_CHARS_PER_CHUNK)

    total_chars = len(script_text)
    print(f"[Episode] {episode_id}: total_chars={total_chars}, chunks={len(chunks)}, max_chars_per_chunk={MAX_CHARS_PER_CHUNK}")

    for idx, c in enumerate(chunks):
        print(f"[Episode] Chunk {idx+1}/{len(chunks)}: chars={len(c)} preview='{_preview(c)}'")

    print(f"[Episode] {episode_id}: starting TTS for {len(chunks)} chunks.")

    wav_segments: List[np.ndarray] = []
    part_idx = 0

    for idx, chunk in enumerate(chunks):
        print(f"[Episode] Generating chunk {idx+1}/{len(chunks)} (len={len(chunk)} chars)")

        # IMPORTANT: pauses do NOT go through TTS
        if is_pause_tag(chunk):
            secs = pause_seconds(chunk, default_s=DEFAULT_PAUSE_S)
            n = int(secs * SAMPLE_RATE)
            if n > 0:
                print(f"[Episode] Inserting pause: {secs:.2f}s")
                wav_segments.append(np.zeros(n, dtype=np.float32))
            continue

        out_part = os.path.join(EPISODE_OUT_DIR, f"{episode_id}_part{part_idx:03d}.wav")
        part_idx += 1

        synthesize_marcus(
            text=chunk,
            out_path=out_part,
            quality=quality,
            use_ref_audio=use_ref_audio,
        )

        audio, sr = sf.read(out_part, always_2d=False)
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"Sample rate mismatch: {sr} != {SAMPLE_RATE} in {out_part}")
        if audio.ndim > 1:
            audio = audio[:, 0]
        wav_segments.append(audio.astype(np.float32))

    if not wav_segments:
        raise RuntimeError("No audio generated (all chunks empty?)")

    final_wav = np.concatenate(wav_segments, axis=0)
    out_final = os.path.join(EPISODE_OUT_DIR, f"{episode_id}.wav")
    sf.write(out_final, final_wav, SAMPLE_RATE)

    print(f"[Episode] Saved final episode: {out_final} ({len(final_wav)/SAMPLE_RATE:.2f} s)")
    return out_final

# ---------------------------------------------------------
# API endpoints
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "sample_rate": SAMPLE_RATE}

@app.post("/tts/line")
async def tts_line(text: str = Form(...), quality: str = Form("fast")):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")

    episode_id = uuid.uuid4().hex[:8]
    out_path = os.path.join(EPISODE_OUT_DIR, f"{episode_id}_line.wav")
    synthesize_marcus(text=text, out_path=out_path, quality=quality, use_ref_audio=True)

    return FileResponse(
        path=out_path,
        media_type="audio/wav",
        filename=os.path.basename(out_path),
    )

@app.post("/episode/from-file")
async def episode_from_file(
    script: UploadFile = File(..., description="Text file containing the full episode script."),
    episode_id: Optional[str] = Form(None, description="Optional episode id / name; if omitted, a random id is generated."),
    quality: str = Form("vivid", description="fast | vivid | max"),
    use_ref_audio: bool = Form(True, description="Use Marcus reference audio for voice cloning"),
):
    if script.content_type not in ("text/plain", "application/octet-stream"):
        raise HTTPException(status_code=400, detail=f"Unsupported content_type: {script.content_type}")

    raw = await script.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded script is empty.")

    print(f"[Episode] {episode_id or 'NEW'}: received script upload, chars={len(text)}, content_type={script.content_type}")

    out_final = build_episode_from_text(text, episode_id=episode_id, quality=quality, use_ref_audio=use_ref_audio)

    return FileResponse(
        path=out_final,
        media_type="audio/wav",
        filename=os.path.basename(out_final),
    )

@app.get("/episode/list")
def list_episodes():
    files = [
        f for f in os.listdir(EPISODE_OUT_DIR)
        if f.lower().endswith(".wav") and "_part" not in f
    ]
    return JSONResponse({"episodes": sorted(files)})

@app.get("/episode/download/{episode_file}")
def download_episode(episode_file: str):
    path = os.path.join(EPISODE_OUT_DIR, episode_file)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Episode not found.")
    return FileResponse(path=path, media_type="audio/wav", filename=episode_file)
