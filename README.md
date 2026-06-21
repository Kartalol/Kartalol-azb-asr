# South Azerbaijani ASR Benchmark

Official training and evaluation code for **South Azerbaijani Automatic Speech Recognition (ASR)**.
This repository provides scripts for fine-tuning ASR models, evaluating benchmark datasets, and reproducing reported WER/CER/SER results.

## Resources

| Resource       | Link                                                             |
| -------------- | ---------------------------------------------------------------- |
| Model          | `https://huggingface.co/Kartal-Ol/ASR-AZB`                                     |
| Dataset        | `https://huggingface.co/datasets/Kartal-Ol/azb-asr-corpus`       |
| Gold benchmark | `https://huggingface.co/datasets/Kartal-Ol/AZB-ASR-Gold-Testset` |
| Paper          | `Preserving the Iranian Turkic Language: Community-Driven ASR Datasets and Benchmarking for South Azerbaijani`                                                    |

## Benchmarking

We evaluate South Azerbaijani ASR models using three metrics:

* **WER (%)**: Word Error Rate
* **CER (%)**: Character Error Rate
* **DIR**: Deletion/Insertion Ratio, reported as a raw ratio

Lower **WER** and **CER** indicate better recognition performance. DIR is reported to describe the error profile and is not used as the primary ranking metric.

### Evaluation Sets

| Test Set  | Description                                                    |
| --------- | -------------------------------------------------------------- |
| External  | Azerbaijani external test data                                 |
| Community | Fully normalized community-curated South Azerbaijani test data |
| GoldSet   | Released South Azerbaijani benchmark set                       |

### Training Regimes

| Training Setup | Description                                                                |
| -------------- | -------------------------------------------------------------------------- |
| Community-only | Trained only on the normalized community-curated South Azerbaijani dataset |
| Full dataset   | Trained on the combination of community-curated and external data sources  |
| No Fine-Tune   | Evaluated without task-specific fine-tuning                                |

## Main Benchmark

Entries are reported as **WER / CER / DIR**.

| Model        | Training Setup |               External |              Community |                GoldSet |
| ------------ | -------------- | ---------------------: | ---------------------: | ---------------------: |
| Whisper-Tiny | Full dataset   |    121.0 / 67.0 / 0.31 |     53.0 / 32.0 / 0.29 |     77.0 / 34.0 / 0.24 |
| Whisper-Tiny | Community-only |    140.0 / 90.0 / 0.48 |     39.0 / 16.0 / 4.15 |     84.0 / 38.0 / 0.25 |
| Whisper-Base | Full dataset   | **64.0 / 44.0 / 0.14** |     49.0 / 41.0 / 0.27 |     70.0 / 28.0 / 0.19 |
| Whisper-Base | Community-only |    122.0 / 68.0 / 0.44 | **33.0 / 14.0 / 2.67** |     84.0 / 36.0 / 0.25 |
| MMS          | No Fine-Tune   |    106.0 / 50.0 / 0.15 | 70.0 / 29.0 / **0.20** | 73.0 / 24.0 / **0.08** |
| MMS          | Community-only |                      — |     50.0 / 19.0 / 0.21 | **63.0 / 18.0 / 0.13** |

## Community-Only Model Comparison

All models below were trained exclusively using the **Community-only** regime.

Entries are reported as **WER / CER / DIR**.

| Model                           |                External |              Community |                GoldSet |
| ------------------------------- | ----------------------: | ---------------------: | ---------------------: |
| Whisper-Tiny                    |     140.0 / 90.0 / 0.48 |     39.0 / 16.0 / 4.15 |     84.0 / 38.0 / 0.25 |
| Whisper-Base                    |     122.0 / 68.0 / 0.44 |     33.0 / 14.0 / 2.67 |     84.0 / 36.0 / 0.25 |
| Whisper-Small                   |     149.0 / 99.0 / 0.38 |     35.0 / 16.0 / 1.47 |     79.0 / 30.0 / 0.21 |
| Whisper-Small-Farsi             | 136.0 / 76.0 / **0.32** |     31.0 / 13.0 / 2.44 |     79.0 / 31.0 / 0.22 |
| Whisper-Small-North Azerbaijani |     130.0 / 74.0 / 0.35 | **22.0 / 13.0 / 2.60** |     82.0 / 33.0 / 0.24 |
| Whisper-Small-Turkish           |     159.0 / 96.0 / 0.44 | 29.0 / **12.0** / 2.48 |     79.0 / 29.0 / 0.21 |
| Whisper-Small-Arabic            |     152.0 / 96.0 / 0.50 |     43.0 / 24.0 / 3.62 |     83.0 / 42.0 / 0.37 |
| MMS                             |                       — | 50.0 / 20.0 / **0.21** | **63.0 / 18.0 / 0.13** |

## Key Observations

* On the **Community** test set, the best WER is achieved by **Whisper-Small-North Azerbaijani** with **22.0% WER**.
* On the **GoldSet**, **MMS** achieves the best overall result with **63.0% WER** and **18.0% CER**.
* Among Whisper-based models on the GoldSet, **Whisper-Tiny trained on the full dataset** achieves the lowest WER, with **77.0% WER**.
* Cross-lingual fine-tuning with related languages such as **North Azerbaijani**, **Turkish**, and **Farsi** improves performance on the community-curated test set compared with several general Whisper baselines.
* The gap between Community and GoldSet results shows that the GoldSet is more challenging and better reflects broader benchmark conditions.

## Reproducing Evaluation

```bash
python evaluate.py \
  --model_name YOUR_MODEL_OR_CHECKPOINT \
  --dataset_name Kartal-Ol/AZB-ASR-Gold-Testset \
  --metrics wer cer dir
```

## Hugging Face Resources

| Resource          | Link                                                             |
| ----------------- | ---------------------------------------------------------------- |
| ASR corpus        | `https://huggingface.co/datasets/Kartal-Ol/azb-asr-corpus`       |
| GoldSet benchmark | `https://huggingface.co/datasets/Kartal-Ol/AZB-ASR-Gold-Testset` |
| Model             | `https://huggingface.co/YOUR_MODEL_NAME`                         |


## Installation

```bash
git clone https://github.com/YOUR_USERNAME/south-azerbaijani-asr-benchmark.git
cd south-azerbaijani-asr-benchmark
pip install -r requirements.txt
```

## Dataset

The training and evaluation datasets are hosted on Hugging Face.

```python
from datasets import load_dataset

dataset = load_dataset("Kartal-Ol/azb-asr-corpus")
```

## Training

```bash
python train.py \
  --dataset_name Kartal-Ol/azb-asr-corpus \
  --model_name facebook/w2v-bert-2.0 \
  --output_dir outputs/azb-asr
```

## Evaluation

```bash
python evaluate.py \
  --model_name YOUR_HF_MODEL_NAME \
  --dataset_name Kartal-Ol/AZB-ASR-Gold-Testset
```

## Language Model Decoding

```bash
python decode_with_kenlm.py \
  --model_name YOUR_HF_MODEL_NAME \
  --kenlm_model path/to/azb_5gram.arpa \
  --dataset_name Kartal-Ol/AZB-ASR-Gold-Testset
```

## Repository Structure

```text
.
├── train.py
├── evaluate.py
├── decode_with_kenlm.py
├── requirements.txt
├── configs/
├── scripts/
├── examples/
└── README.md
```

## Citation

If you use this code, model, or dataset, please cite:

```bibtex
@article{your2026southazerbaijaniasr,
  title={Preserving the Iranian Turkic Language: A Community-Based ASR Dataset and Model Evaluation for South Azerbaijani},
  author={...},
  year={2026}
}
```

## License

This code is released under the MIT License.
The datasets and models are released under their respective Hugging Face licenses.
