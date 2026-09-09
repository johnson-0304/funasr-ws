# funasr-ws

在纯 CPU 上运行 **Fun-ASR-Nano**（800M，中英日 + 中文方言）的语音识别服务，同时提供：

- **WebSocket**：兼容 FunASR runtime 的 `2pass` 协议，现有 FunASR 客户端（含 Jarvis）可直接连接，边说边出中间结果，句末给最终结果。
- **HTTP**：OpenAI 兼容的 `POST /v1/audio/transcriptions`，整段文件转写。

推理用 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 的官方 int8 ONNX 模型，句子切分用 Silero VAD。模型在构建镜像时打入，容器启动即可用，无需 GPU、无需 PyTorch。

## 运行

```bash
docker run -d --name funasr-ws -p 127.0.0.1:10097:10095 \
  -e NUM_THREADS=4 ghcr.io/johnson-0304/funasr-ws:latest
curl http://127.0.0.1:10097/health
```

或 `docker compose up -d`（见 `docker-compose.yml`）。首次加载模型约 3–5 秒，`/health` 返回 200 即就绪。

### 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `NUM_THREADS` | `4` | 识别模型线程数。大小核混合的 Intel CPU 上 4 比 8 快 |
| `ASR_LANGUAGE` | 空 | 固定语言 `zh` / `en` / `ja`；空为自动识别（中英混说也可） |
| `ASR_ITN` | `true` | 数字、时间等逆文本规整 |
| `VAD_THRESHOLD` | `0.5` | Silero VAD 语音阈值，越高越不易被噪声触发 |
| `VAD_MIN_SILENCE_S` | `0.4` | 静音多久算一句结束 |
| `VAD_MAX_SPEECH_S` | `15` | 单句最长秒数，超过强制切分 |
| `PARTIAL_INTERVAL_MS` | `800` | 中间结果重解码间隔 |
| `PRE_ROLL_MS` / `POST_ROLL_MS` | `300` / `150` | 句段前后补的音频，避免句首弱音被 VAD 切掉 |
| `MAX_HTTP_AUDIO_S` | `600` | HTTP 上传音频最长秒数 |
| `HOST` / `PORT` | `0.0.0.0` / `10095` | 监听地址 |
| `MODEL_DIR` | `/models` | 模型目录，可挂载覆盖 |

## WebSocket 协议

路径 `/` 或 `/ws`。先发一条 JSON 起始帧，再发 16 kHz、16 bit、单声道 PCM 二进制帧，说完发 `{"is_speaking": false}`。

```json
{"mode": "2pass", "wav_name": "demo", "wav_format": "pcm", "audio_fs": 16000,
 "is_speaking": true, "svs_lang": "auto", "speech_noise_thres": 0.5}
```

服务端返回：

| 帧 | 含义 |
| --- | --- |
| `{"mode":"config","text":"","speech_noise_thres":0.5}` | 起始帧带 `speech_noise_thres` 时的确认；该值作为本连接的 Silero VAD 阈值 |
| `{"mode":"2pass-online","text":"...","is_final":false}` | 当前句新增的文字（增量） |
| `{"mode":"2pass-offline","text":"...","is_final":false}` | 一句结束后的完整文本；`is_speaking:false` 触发的最后一句 `is_final` 为 `true` |
| `{"mode":"config","text":"","error":"..."}` | 协议错误，随后关闭连接（1008） |

`svs_lang` 接受 `auto` / `zh` / `en` / `ja`。当前 sherpa-onnx 版本不支持按连接切换语言与热词，服务会记录警告并沿用 `ASR_LANGUAGE`；需要固定语言时用环境变量。

## HTTP API

```bash
curl http://127.0.0.1:10097/v1/audio/transcriptions -F file=@audio.wav
curl http://127.0.0.1:10097/v1/audio/transcriptions -F file=@audio.wav \
  -F response_format=verbose_json -F language=zh
```

- `file`：wav / flac / ogg 等 libsndfile 支持的格式，任意采样率，自动转 16 kHz 单声道。
- `response_format`：`json`（默认）、`text`、`verbose_json`（含每句起止时间）。
- `language`：`zh` / `en` / `ja`，省略为自动。
- `GET /health`、`GET /v1/models`。

## 本地开发

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
scripts/download-models.sh models        # MODEL_MIRROR=modelscope 可走国内镜像
MODEL_DIR=models python -m funasr_ws
ruff check . && pytest
```

## 实测（i5-13500H，纯 CPU，4 线程）

Kokoro 合成的 16 kHz 中英测试音频：8 秒中文单句解码约 1.4 秒（RTF ≈ 0.17）；实时推流时说话开始约 1.2 秒后出第一条中间结果，句末约 1 秒后出最终结果。中文、英文、中英混说均逐字正确；同一批音频 SenseVoiceSmall 出现 `Gitthub`、`jascript` 等错误。

## CI / 发布

- `Checks`：push / PR 时跑 ruff 与 pytest。
- `Build and Publish Docker Image`：push 到 `main` 或打 `v*` tag 时构建并推送 `ghcr.io/<owner>/funasr-ws`，完成后调用通知 webhook。
