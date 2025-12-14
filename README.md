## voxcpm-1.5-fastapi-server
DGX Spark–focused example: Dockerized FastAPI REST server for VoxCPM 1.5 TTS (voice prompting + batch text files).

## License
This repo is MIT licensed; VoxCPM and the NVIDIA base container are licensed separately.

## Build

The commands below assume you want your local checkout to live at `~/Projects/VoxCPM-API` so it matches the paths used in the **Usage** examples.

```bash
mkdir -p ~/Projects/VoxCPM-API
cd ~/Projects/VoxCPM-API
git clone https://github.com/operezmuena/voxcpm-1.5-fastapi-server.git .
docker build -t voxcpm-api:latest .
```

## Usage

All examples below start the API on port **8000** and mount:

* `~/Projects/VoxCPM-API/episodes` → `/workspace/episodes`
* `~/.cache/huggingface` → `/root/.cache/huggingface` (model cache)

### 1) Basic

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --name voxcpm-api \
  -p 8000:8000 \
  -v ~/Projects/VoxCPM-API/episodes:/workspace/episodes \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  voxcpm-api:latest
```

### 2) With tuning knobs (useful for testing cadence/pacing)

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --name voxcpm-api \
  -p 8000:8000 \
  -e MARCUS_CAP_INTERNAL_SILENCE=1 \
  -e MARCUS_INTERNAL_MAX_SILENCE_S=0.45 \
  -e MARCUS_INTERNAL_TARGET_SILENCE_S=0.16 \
  -e MARCUS_INTERNAL_REL_THRESH=0.006 \
  -v ~/Projects/VoxCPM-API/episodes:/workspace/episodes \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  voxcpm-api:latest
```

### 3) Using an environment file (`marcus_voxcpm.env`)

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --name voxcpm-api \
  -p 8000:8000 \
  --env-file ~/Projects/VoxCPM-API/marcus_voxcpm.env \
  -v ~/Projects/VoxCPM-API/episodes:/workspace/episodes \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  voxcpm-api:latest
```

Example `marcus_voxcpm.env`:

```bash
MARCUS_CAP_INTERNAL_SILENCE=1
MARCUS_INTERNAL_MAX_SILENCE_S=0.45
MARCUS_INTERNAL_TARGET_SILENCE_S=0.16
```

## Verify the server is running

Once the container is up, you should be able to hit the FastAPI docs:

```bash
curl -I http://localhost:8000/docs
```

You can also check the OpenAPI JSON:

```bash
curl http://localhost:8000/openapi.json | head
```

## Example POST (generate audio from a text file)

```bash
curl -X POST "http://localhost:8000/episode/from-file" \
  -F "episode_id=213_ep1_b1" \
  -F "script=@/path/to/213_ep1_b1.txt;type=text/plain"
```

After the request completes, the generated WAV files will be written to the mounted output directory (e.g., `~/Projects/VoxCPM-API/episodes`).
