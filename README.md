# South Azerbaijani ASR Benchmark

Official research repository for the INTERSPEECH 2026 paper **“Preserving the
Iranian Turkic Language: Community-Driven ASR Datasets and Benchmarking for
South Azerbaijani.”** It provides reproducible dataset loading, South
Azerbaijani text normalization, Whisper and MMS fine-tuning, Whisper/MMS/CTC
inference, and WER/CER/DIR benchmark evaluation without storing large audio or
model files in Git.

## Contributions

- Community-driven South Azerbaijani speech resources in Perso-Arabic script.
- Evaluation across Community, External, and held-out GoldSet data.
- Whisper Tiny/Base/Small, cross-lingual Whisper, and MMS-1B-All experiments.
- A single explicit evaluation protocol for WER, CER, and deletion/insertion
  ratio (DIR).

## Datasets

| Role | Resource | Use |
|---|---|---|
| Corpus/index | [Kartal-Ol/azb-asr-corpus](https://huggingface.co/datasets/Kartal-Ol/azb-asr-corpus) | Community/external transcript resources |
| GoldSet | [Kartal-Ol/AZB-ASR-Gold-Testset](https://huggingface.co/datasets/Kartal-Ol/AZB-ASR-Gold-Testset) | Evaluation only |
| External | [BHOSAI/PseudoLabelled_Azerbaijani_Voices](https://huggingface.co/datasets/BHOSAI/PseudoLabelled_Azerbaijani_Voices) | Full-dataset training component |
| External | [VoxLingua107](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/) Azerbaijani | Full-dataset training component |

All source rows are converted internally to `audio`, `text`, and optional
`speaker_id`/`duration`. GoldSet is programmatically rejected by training
commands. See [dataset roles and known schema TODOs](docs/datasets.md).

## Models

The unified Whisper trainer accepts `openai/whisper-tiny`,
`openai/whisper-base`, `openai/whisper-small`, or any compatible Hugging Face
Whisper checkpoint. This is also how the Farsi, North Azerbaijani, Turkish, and
Arabic cross-lingual initializations are supplied. The MMS trainer activates the
`azb` adapter of `facebook/mms-1b-all` and trains the adapter and CTC head by
default. Public paper checkpoints are collected at
[Kartal-Ol/ASR-AZB](https://huggingface.co/Kartal-Ol/ASR-AZB).

### Hugging Face model downloads

| Model | Checkpoint files |
|---|---|
| Whisper Tiny | [Kartal-Ol/ASR-AZB — whisper-tiny](https://huggingface.co/Kartal-Ol/ASR-AZB/tree/main/whisper-tiny) |
| Whisper Base | [Kartal-Ol/ASR-AZB — whisper-base](https://huggingface.co/Kartal-Ol/ASR-AZB/tree/main/whisper-base) |
| Whisper Base — Full dataset | [Kartal-Ol/ASR-AZB — whisper-base-full](https://huggingface.co/Kartal-Ol/ASR-AZB/tree/main/whisper-base-full) |
| Whisper Small | [Kartal-Ol/ASR-AZB — whisper-Small](https://huggingface.co/Kartal-Ol/ASR-AZB/tree/main/whisper-Small) |
| Whisper Small — Farsi | [Kartal-Ol/ASR-AZB — whisper-Small-Farsi](https://huggingface.co/Kartal-Ol/ASR-AZB/tree/main/whisper-Small-Farsi) |
| Whisper Small — North Azerbaijani | [Kartal-Ol/ASR-AZB — whisper-small-north-azerbaijani](https://huggingface.co/Kartal-Ol/ASR-AZB/tree/main/whisper-small-north-azerbaijani) |
| Whisper Small — Turkish | [Kartal-Ol/ASR-AZB — whisper-small-turkish](https://huggingface.co/Kartal-Ol/ASR-AZB/tree/main/whisper-small-turkish) |
| Whisper Small — Arabic | [Kartal-Ol/ASR-AZB — whisper-Small-Arabic](https://huggingface.co/Kartal-Ol/ASR-AZB/tree/main/whisper-Small-Arabic) |
| MMS | [Kartal-Ol/ASR-AZB — mms](https://huggingface.co/Kartal-Ol/ASR-AZB/tree/main/mms) |

The links point directly to each checkpoint directory so model weights,
processor/tokenizer files, and configuration files can be downloaded together.

### Google Colab inference workshop

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kartalol/Kartalol-azb-asr/blob/code_refactor/docs/colab_inference_demo.ipynb)

The [workshop notebook](docs/colab_inference_demo.ipynb) loads any published
Whisper checkpoint directly from Hugging Face, accepts an uploaded audio sample,
and provides one-sample WER, CER, DIR, and auxiliary SER analysis with a word
alignment. SER is a tutorial diagnostic and is not substituted for DIR in the
paper benchmark.

## Benchmark results

WER and CER are percentages. DIR is the raw word-level deletion/insertion
ratio, `deletions / insertions`. Lower WER and CER are better. These values are
transcribed unchanged from the repository’s paper README; they have not been
recomputed during cleanup.

| Model | Training Setup | External | Community | GoldSet |
|---|---|---:|---:|---:|
| Whisper-Tiny | Full dataset | 121.0 / 67.0 / 0.31 | 53.0 / 32.0 / 0.29 | 77.0 / 34.0 / 0.24 |
| Whisper-Tiny | Community-only | 140.0 / 90.0 / 0.48 | 39.0 / 16.0 / 4.15 | 84.0 / 38.0 / 0.25 |
| Whisper-Base | Full dataset | **64.0 / 44.0 / 0.14** | 49.0 / 41.0 / 0.27 | 70.0 / 28.0 / 0.19 |
| Whisper-Base | Community-only | 122.0 / 68.0 / 0.44 | **33.0 / 14.0 / 2.67** | 84.0 / 36.0 / 0.25 |
| MMS | No Fine-Tune | 106.0 / 50.0 / 0.15 | 70.0 / 29.0 / **0.20** | 73.0 / 24.0 / **0.08** |
| MMS | Community-only | — | 50.0 / 19.0 / 0.21 | **63.0 / 18.0 / 0.13** |

Community-only cross-lingual comparison (WER / CER / DIR):

| Model | External | Community | GoldSet |
|---|---:|---:|---:|
| Whisper-Small | 149.0 / 99.0 / 0.38 | 35.0 / 16.0 / 1.47 | 79.0 / 30.0 / 0.21 |
| Whisper-Small-Farsi | 136.0 / 76.0 / 0.32 | 31.0 / 13.0 / 2.44 | 79.0 / 31.0 / 0.22 |
| Whisper-Small-North Azerbaijani | 130.0 / 74.0 / 0.35 | 22.0 / 13.0 / 2.60 | 82.0 / 33.0 / 0.24 |
| Whisper-Small-Turkish | 159.0 / 96.0 / 0.44 | 29.0 / 12.0 / 2.48 | 79.0 / 29.0 / 0.21 |
| Whisper-Small-Arabic | 152.0 / 96.0 / 0.50 | 43.0 / 24.0 / 3.62 | 83.0 / 42.0 / 0.37 |

## Installation

Python 3.10 or newer is required. Audio decoding for MP3/OGG may additionally
require an FFmpeg installation supported by the local audio backend.

```bash
git clone https://github.com/Kartalol/Kartalol-azb-asr.git
cd Kartalol-azb-asr
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Inference

Whisper, MMS, and compatible Wav2Vec2/CTC checkpoints are detected from their
Transformers configuration. Input is converted to mono and resampled to the
processor’s required rate.

```bash
python scripts/transcribe.py \
  --model openai/whisper-tiny \
  --audio example.ogg
```

Use `--device cpu`, `--device cuda`, or `--device cuda:1` to override automatic
selection. Batch inference is available with `--folder AUDIO_DIRECTORY`; add
`--recursive` if needed and `--output-json outputs/transcriptions.json` for
machine-readable output.

## Training

The checked-in configurations preserve parameters recoverable from the old
scripts. Every inferred reproducibility default is marked `TODO` in YAML.

```bash
python scripts/train_whisper.py --config configs/whisper_tiny_community.yaml
python scripts/train_whisper.py --config configs/whisper_base_full.yaml
python scripts/train_mms.py --config configs/mms_community.yaml --fp16
```

For a cross-lingual Whisper experiment, override the compatible checkpoint and
output directory without editing source:

```bash
python scripts/train_whisper.py \
  --config configs/whisper_small_community.yaml \
  --model-id YOUR_COMPATIBLE_WHISPER_CHECKPOINT \
  --output-dir outputs/whisper-small-cross-lingual
```

Available paper configurations:

- `configs/whisper_tiny_community.yaml`
- `configs/whisper_tiny_full.yaml`
- `configs/whisper_base_community.yaml`
- `configs/whisper_base_full.yaml`
- `configs/whisper_small_community.yaml`
- `configs/mms_community.yaml`

## Evaluation

Official scores use raw references and predictions; normalization is never
enabled silently.

```bash
python scripts/evaluate.py \
  --model YOUR_MODEL_OR_HF_ID \
  --dataset goldset \
  --split test \
  --normalization none \
  --output-dir outputs/goldset
```

This writes `metrics.json` and `predictions.csv`. Use `--dataset community` or
`--dataset external` for the other benchmark sets. An additional normalized
analysis can use `--normalization published`; see
[normalization documentation](docs/normalization.md).

## Reproducing the benchmark

Copy the example manifest, replace placeholder checkpoints, and run:

```bash
cp configs/benchmark.example.yaml configs/benchmark.local.yaml
python scripts/reproduce_benchmark.py \
  --manifest configs/benchmark.local.yaml \
  --output-dir outputs/benchmark
```

The command evaluates each supplied checkpoint on External, Community, and
GoldSet and writes CSV, JSON, and Markdown tables in the paper format. Exact
reproduction still requires the final checkpoint IDs, exact Hugging Face
External split/config, package/environment lock, and confirmation of the
original random seeds and generation settings; these were not recoverable from
the checked-in experiment scripts.

## Repository structure

```text
configs/                 Paper experiment and benchmark manifests
docs/                    Dataset and normalization protocol
legacy/                  Sanitized historical experiment notes
scripts/                 Training, inference, evaluation, reproduction CLIs
src/kartalol_azb_asr/    Reusable data, audio, normalization, metrics, inference
tests/                   Lightweight unit tests (no model downloads)
```

## Citation

```bibtex
@inproceedings{farsi2026preserving,
  title     = {Preserving the Iranian Turkic Language: Community-Driven ASR
               Datasets and Benchmarking for South Azerbaijani},
  author    = {Farsi, Farhan and Bali, Shayan and Nourmohammadi Khiarak, Jalil
               and Aref, Mohammad Hossein and Akbari Saeed, Taher},
  booktitle = {Proceedings of INTERSPEECH 2026},
  year      = {2026}
}
```

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff).

## License and acknowledgements

Code is released under the existing [MIT License](LICENSE). Datasets and models
retain the licenses shown on their respective hosting pages.

We thank the Kartal Ol Foundation and every community contributor whose speech,
transcription, review, and organizational work made these resources possible.
