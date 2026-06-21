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

## Benchmark Results

| Model                  |  Dataset |    WER |    CER |    SER |
| ---------------------- | -------: | -----: | -----: | -----: |
| wav2vec2-BERT          | Test set | 0.1770 | 0.0433 | 0.2989 |
| wav2vec2-BERT + KenLM  | Test set | 0.1372 | 0.0233 | 0.2189 |
| QuartzNet15x5          | Test set | 0.5502 | 0.2507 | 0.7362 |
| Whisper Large V3 Turbo | Test set | 1.1370 | 0.8330 | 0.9189 |

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
