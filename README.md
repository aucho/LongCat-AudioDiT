# LongCat-AudioDiT: High-Fidelity Diffusion Text-to-Speech in the Waveform Latent Space

<div align="center">
  <img src="assets/LongCat-AudioDiT.svg" width="45%" alt="LongCat-AudioDiT" />
</div>
<hr>

<div align="center" style="line-height: 1;">
    <a href="https://arxiv.org/abs/2603.29339">
    <img alt="Paper" src="https://img.shields.io/badge/arXiv-2603.29339-b31b1b.svg" style="display: inline-block; vertical-align: middle;"/>  
    </a>
    <a href="https://github.com/meituan-longcat/LongCat-AudioDiT" target="_blank" style="margin: 2px;">
        <img alt="GitHub" src="https://img.shields.io/badge/GitHub-LongCatAudioDiT-white?logo=github&logoColor=white&color=a4b5d5" style="display: inline-block; vertical-align: middle;"/>
    </a>
        <a href="https://aria-k-alethia.github.io/LongCat-AudioDiT-demo" target="_blank" style="margin: 2px;">
        <img alt="Demo" src="https://img.shields.io/badge/Demo-LongCatAudioDiT-white?logo=googleplay&logoColor=white&color=eabcdd" style="display: inline-block; vertical-align: middle;"/>
    </a>
</div>
<div align="center" style="line-height: 1;">
    <a href="https://huggingface.co/meituan-longcat/LongCat-AudioDiT-3.5B" target="_blank" style="margin: 2px;">
        <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LongCatAudioDiT3.5B-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://huggingface.co/meituan-longcat/LongCat-AudioDiT-1B" target="_blank" style="margin: 2px;">
        <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LongCatAudioDiT1B-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://github.com/meituan-longcat/LongCat-AudioDiT/blob/main/assets/wechat_official_accounts.png" target="_blank" style="margin: 2px;">
    <img alt="Wechat" src="https://img.shields.io/badge/WeChat-LongCat-brightgreen?logo=wechat&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://x.com/Meituan_LongCat" target="_blank" style="margin: 2px;">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-LongCat-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
    <a href="https://github.com/meituan-longcat/LongCat-AudioDiT/blob/main/LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

## Introduction

LongCat-AudioDiT is a state-of-the-art (SOTA) diffusion-based text-to-speech (TTS) model that directly operates in the waveform latent space.
> **Abstract**: We present LongCat-TTS, a novel, non-autoregressive diffusion-based text-to-speech (TTS) model that achieves state-of-the-art (SOTA) performance.
Unlike previous methods that rely on intermediate acoustic representations such as mel-spectrograms, the core innovation of LongCat-TTS lies in operating directly within the waveform latent space. This approach effectively mitigates compounding errors and drastically simplifies the TTS pipeline, requiring only a waveform variational autoencoder (Wav-VAE) and a diffusion backbone.
Furthermore, we introduce two critical improvements to the inference process: first, we identify and rectify a long-standing training-inference mismatch; second, we replace traditional classifier-free guidance with adaptive projection guidance to elevate generation quality.
Experimental results demonstrate that, despite the absence of complex multi-stage training pipelines or high-quality human-annotated datasets, LongCat-TTS achieves SOTA zero-shot voice cloning performance on the Seed benchmark while maintaining competitive intelligibility.
Specifically, our largest variant, LongCat-TTS-3.5B, outperforms the previous SOTA model (Seed-TTS), improving the speaker similarity (SIM) scores from 0.809 to 0.818 on Seed-ZH, and from 0.776 to 0.797 on Seed-Hard.
Finally, through comprehensive ablation studies and systematic analysis, we validate the effectiveness of our proposed modules.
Notably, we investigate the interplay between the Wav-VAE and the TTS backbone, revealing the counterintuitive finding that superior reconstruction fidelity in the Wav-VAE does not necessarily lead to better overall TTS performance.
Code and model weights are released to foster further research within the speech community.

![image](assets/architecture.png)

This repository provides the HuggingFace-compatible implementation, including model definition, weight conversion, and inference scripts.

## Experimental Results on Seed Benchmark
LongCat-AudioDiT obtains state-of-the-art (SOTA) voice cloning performance on the Seed-benchmark, surpassing both close-source and open-source modles.

| **Model** | **ZH CER (%)** ↓ | **ZH SIM** ↑ | **EN WER (%)** ↓ | **EN SIM** ↑ | **ZH-Hard CER (%)** ↓ | **ZH-Hard SIM** ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| GT | 1.26 | 0.755 | 2.14 | 0.734 | - | - |
| Seed-DiT | 1.18 | 0.809 | 1.73 | **0.790** | - | - |
| MaskGCT | 2.27 | 0.774 | 2.62 | 0.714 | 10.27 | 0.748 |
| E2 TTS | 1.97 | 0.730 | 2.19 | 0.710 | - | - |
| F5 TTS | 1.56 | 0.741 | 1.83 | 0.647 | 8.67 | 0.713 |
| F5R-TTS | 1.37 | 0.754 | - | - | 8.79 | 0.718 |
| ZipVoice | 1.40 | 0.751 | 1.64 | 0.668 | - | - |
| Seed-ICL | 1.12 | 0.796 | 2.25 | 0.762 | 7.59 | 0.776 |
| SparkTTS | 1.20 | 0.672 | 1.98 | 0.584 | - | - |
| FireRedTTS | 1.51 | 0.635 | 3.82 | 0.460 | 17.45 | 0.621 |
| Qwen2.5-Omni | 1.70 | 0.752 | 2.72 | 0.632 | 7.97 | 0.747 |
| Qwen2.5-Omni_RL | 1.42 | 0.754 | 2.33 | 0.641 | 6.54 | 0.752 |
| CosyVoice | 3.63 | 0.723 | 4.29 | 0.609 | 11.75 | 0.709 |
| CosyVoice2 | 1.45 | 0.748 | 2.57 | 0.652 | 6.83 | 0.724 |
| FireRedTTS-1S | 1.05 | 0.750 | 2.17 | 0.660 | 7.63 | 0.748 |
| CosyVoice3-1.5B | 1.12 | 0.781 | 2.21 | 0.720 | *5.83* | 0.758 |
| IndexTTS2 | 1.03 | 0.765 | 2.23 | 0.706 | 7.12 | 0.755 |
| DiTAR | 1.02 | 0.753 | 1.69 | 0.735 | - | - |
| MiniMax-Speech | 0.99 | 0.799 | 1.90 | 0.738 | - | - |
| VoxCPM | *0.93* | 0.772 | 1.85 | 0.729 | 8.87 | 0.730 |
| MOSS-TTS | 1.20 | 0.788 | 1.85 | 0.734 | - | - |
| Qwen3-TTS | 1.22 | 0.770 | **1.23** | 0.717 | 6.76 | 0.748 |
| CosyVoice3.5 | **0.87** | 0.797 | 1.57 | 0.738 | **5.71** | 0.786 |
| LongCat-AudioDiT-1B | 1.18 | *0.812* | 1.78 | 0.762 | 6.33 | *0.787* |
| LongCat-AudioDiT-3.5B | 1.09 | **0.818** | *1.50* | *0.786* | 6.04 | **0.797** |

*Notes*:

1. Results of MOSS-TTS are from [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)
2. Results of CosyVoice3.5 are from [CosyVoice3.5](https://mp.weixin.qq.com/s/sTNC7bVphs9zofly3lBoUQ)

## Installation

```bash
pip install -r requirements.txt
```

## CLI Inference

```bash
# TTS
python inference.py --text "I have 12.5 apples." --language en --output_audio output.wav --model_dir meituan-longcat/LongCat-AudioDiT-3.5B

# Voice cloning
python inference.py \
    --text "I have 12.5 apples." \
    --prompt_text "This is the exact sample transcript." \
    --prompt_audio assets/prompt.wav \
    --output_audio output.wav \
    --model_dir meituan-longcat/LongCat-AudioDiT-3.5B \
    --language en \
    --guidance_method apg

# Batch inference (SeedTTS eval format, one item per line: uid|prompt_text|prompt_wav_path|gen_text)
python batch_inference.py \
    --lst /path/to/meta.lst \
    --output_dir /path/to/output \
    --model_dir meituan-longcat/LongCat-AudioDiT-3.5B \
    --language en \
    --guidance_method apg
```

## Gradio Demo

The Gradio page supports both text-to-speech and zero-shot voice cloning. The
voice-cloning tab requires the uploaded sample audio, its matching transcript,
and the text to synthesize.

```bash
python gradio_app.py
```

The default model is `meituan-longcat/LongCat-AudioDiT-3.5B`; it listens on
`0.0.0.0:7860` for LAN access. Use a local model directory or another compatible
Hugging Face model with `--model_dir`, and select a device or port when needed:

```bash
python gradio_app.py --model_dir /path/to/model --device cuda:0 --port 7860
```

The **Long Text Generation** tab segments English or Spanish text, generates
each segment with the locally loaded model and the same required voice-cloning
sample/transcript, then streams the temporary WAV files through FFmpeg into one
24 kHz mono 192 kbps MP3. It is intended for functional testing; external
inference APIs, queues, and interrupted-job recovery are not included.
Long-text stitching currently adds no synthetic boundary pause and no fade. Both
behaviors are controlled centrally in `app_config.py`.
The Gradio target and maximum segment-duration sliders allow testing up to 180
seconds. During this test phase the service also overrides the model's runtime
duration cap to 180 seconds, then subtracts the reference-audio budget.
For the A20 24 GB deployment profile, the shared defaults target 50-second text
segments with a 60-second maximum. At the default 1.25 speech rate, their actual
generated audio is typically shorter. The 180-second value is only a per-model-
call window; it is not a total MP3 duration limit.

Shared model, generation, segmentation, pause, UI, and CLI defaults are defined
in `app_config.py`. Change that file when tuning test defaults so every entry
point remains consistent.

### English and Spanish number normalization

All Gradio, CLI, batch, and service entry points verbalize common numbers before
duration estimation and tokenization. Examples include `12,000` → `twelve
thousand`, `12.50` → `twelve point five zero`, `12-1` → `twelve dash one`, and
`10.2-inch` → `ten point two inch`. Spanish uses Spanish number words and
`coma`/`guion`. Integers, decimals, signs, ordinals, percentages, and common
length/weight/temperature units are supported.

URLs, email addresses, IP addresses, dates, times, semantic versions, phone
numbers, and mixed product identifiers are preserved because their intended
reading is ambiguous. Normalize those formats explicitly in the calling
application when a particular spoken form is required.

Number verbalization uses the LGPL-licensed `num2words` dependency. Review its
license against your production distribution requirements before release.

### Reusable service API

UI-independent generation workflows live in `services/`; stateless text,
number, audio, segmentation, and FFmpeg helpers live in `utils/`. A future HTTP
adapter can reuse the same loaded service without importing Gradio:

```python
from services import AudioDiTService

service = AudioDiTService.from_pretrained(
    "meituan-longcat/LongCat-AudioDiT-3.5B", device="cuda:0"
)
sample_rate, waveform = service.generate_voice_clone(
    "I have 12.5 apples.",
    "sample.wav",
    "This sample contains 10 words.",
    language="en",
    speech_rate=1.25,
)
```

The default `speech_rate` is `1.25`, which shortens only the generated portion's
target duration and aims for roughly 180 words per minute on typical English or
Spanish prose. Use `1.0` for the checkpoint's original pace. Gradio, CLI and
batch inference expose `speech_rate` for testing.

## Asynchronous API

The API and Gradio entry points are independent. The API loads one model at
startup and processes all generation jobs through a single FIFO worker:

```bash
python api_app.py --device cuda:0 --host 0.0.0.0 --port 7861 \
  --data_dir data/longcat_api --result_ttl_hours 24 --max_pending_tasks 20
```

Only one Uvicorn worker is supported. References are cached by ID and retained;
terminal task manifests and MP3 results are removed 24 hours after completion by
default. Completed results survive service restarts, while interrupted pending
or processing jobs are marked failed and may be submitted again.

The Ubuntu host must provide `ffmpeg` and `ffprobe` on `PATH`. Reference uploads
are streamed to the configured data filesystem, limited to 50 MiB and 60
seconds, validated, and stored once as 24 kHz mono PCM WAV. Reference transcripts
are limited to 10,000 characters. Generation requests accept up to 2,000,000
characters, but there is no estimated-duration or final-MP3 duration limit.

The LAN API contract is:

- `POST /v1/references/add`: multipart fields `id`, `audio`, and exact `text`;
  returns `content_sha256` and `reused`. Reusing an ID with different content
  returns HTTP 409.
- `POST /generate_audio_enhanced_async`: JSON fields `step_id`, `text`,
  `language` (`en` or `es`), `reference_id`, plus optional inference settings.
- `GET /get_task_status?step_id=...`: task state, timestamps, progress and error.
- `GET /download_result?step_id=...`: completed 24 kHz mono 192 kbps MP3.
- `POST /stop_async_task/{step_id}`: pending cancellation or a cancellation
  request observed at the next segment boundary. A processing response includes
  `cancel_requested: true` and must still be polled to a terminal state.
- `GET /v1/health`: model/device readiness, worker and cleaner state, active
  task, real pending count and background errors. It returns HTTP 503 when the
  model or durable worker is not ready.

Resource IDs are portable file-safe identifiers containing only letters,
numbers, `.`, `_`, and `-`. Path separators, `..`, trailing dots, control
characters, and Windows device names are rejected. Persisted result/reference
paths are resolved and constrained to their configured data directories before
every read, download, retry, cleanup, or deletion.

The service is intended for a trusted LAN and does not include authentication.

### Ubuntu systemd service and boot startup

The repository includes a production-oriented single-process systemd unit in
`deploy/systemd/longcat-audiodit.service`. Its default paths assume:

- repository: `/opt/LongCat-AudioDiT`
- Conda environment Python: `/opt/miniconda3/envs/longcat/bin/python`
- service account: `longcat`
- persistent API data: `/var/lib/longcat-audiodit`
- Hugging Face cache: `/var/cache/longcat-audiodit/huggingface`

Change `WorkingDirectory` and `ExecStart` in the unit if the repository or
Conda installation uses different paths. Install and enable it with:

```bash
sudo useradd --system --home-dir /var/lib/longcat-audiodit \
  --shell /usr/sbin/nologin longcat
sudo cp deploy/systemd/longcat-audiodit.env.example /etc/longcat-audiodit.env
sudo cp deploy/systemd/longcat-audiodit.service /etc/systemd/system/
sudo chown root:longcat /etc/longcat-audiodit.env
sudo chmod 0640 /etc/longcat-audiodit.env
sudo systemctl daemon-reload
sudo systemctl enable --now longcat-audiodit.service
```

`enable --now` starts the API immediately and registers it for subsequent
boots. The unit uses one API process and one inference worker, restarts after an
unexpected process exit, waits for network availability, and gives an active
generation up to five minutes to stop cleanly. systemd creates the state and
cache directories with service-account ownership; process logs go to journald.

Check service state, health, and logs with:

```bash
systemctl status longcat-audiodit.service
curl --fail-with-body http://127.0.0.1:7861/v1/health
journalctl -u longcat-audiodit.service -f
```

After changing the environment file, restart the process with
`sudo systemctl restart longcat-audiodit.service`. Model-load failures remain
visible through `/v1/health` as HTTP 503; systemd automatically restarts actual
process crashes, but it does not treat an HTTP 503 as a crashed process.

## Inference (Python API)

### 1. TTS
```python
import audiodit  # auto-registers with transformers
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
import torch, soundfile as sf

# Load model
model = AudioDiTModel.from_pretrained("meituan-longcat/LongCat-AudioDiT-3.5B").to("cuda")
model.vae.to_half()  # VAE runs in fp16 (matching original)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)

# Zero-shot synthesis
inputs = tokenizer(["今天晴暖转阴雨，空气质量优至良，空气相对湿度较低。"], padding="longest", return_tensors="pt")
output = model(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    duration=62,  # latent frames
    steps=16,
    cfg_strength=4.0,
    guidance_method="cfg",  # or "apg"
)
sf.write("output.wav", output.waveform.squeeze().cpu().numpy(), 24000)
```

### 2. Voice Cloning (with prompt audio)

```python
import librosa, torch

# Load prompt audio
audio, _ = librosa.load("assets/prompt.wav", sr=24000, mono=True)
prompt_wav = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)  # (1, 1, T)

# Concatenate prompt_text + gen_text for the text encoder
prompt_text = "小偷却一点也不气馁，继续在抽屉里翻找。"
gen_text = "今天晴暖转阴雨，空气质量优至良，空气相对湿度较低。"
inputs = tokenizer([f"{prompt_text} {gen_text}"], padding="longest", return_tensors="pt")

output = model(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    prompt_audio=prompt_wav,
    duration=138,  # prompt_frames + gen_frames
    steps=16,
    cfg_strength=4.0,
    guidance_method="apg",
)
```

## License Agreement
This repository, including both the model weights and the source code, is released under the **MIT License**.

Any contributions to this repository are licensed under the MIT License, unless otherwise stated. This license does not grant any rights to use Meituan trademarks or patents.

For details, see the [LICENSE](./LICENSE) file.

## Contact
Please contact us at <a href="mailto:longcat-team@meituan.com">longcat-team@meituan.com</a> or open an issue if you have any questions.

#### WeChat Group
<img src=./assets/longcat_wechat_group.jpeg width="200px">
