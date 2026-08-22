# Dataset roles and schemas

The repository never stores the speech corpora. Loaders convert each source to
the internal fields `audio`, `text`, and, when available, `speaker_id` and
`duration`.

| Role | Alias | Hugging Face source | Training permitted |
|---|---|---|---|
| Community | `community` | `yoyo-research-group/south-azerbaijani-asr` | Yes |
| External index | `external` | `Kartal-Ol/azb-asr-corpus` | Yes |
| GoldSet | `goldset` | `Kartal-Ol/AZB-ASR-Gold-Testset` | **No** |
| External component | `bhosai` | `BHOSAI/PseudoLabelled_Azerbaijani_Voices` | Yes |

VoxLingua107 Azerbaijani audio is an external component whose AZB transcript
mapping is documented by `Kartal-Ol/azb-asr-corpus`. Because its downloadable
archive is not itself a Hugging Face dataset with the project schema, prepare or
publish the aligned manifest before passing it as a custom dataset ID.

Dataset hosting schemas can evolve. If automatic field detection fails, pass
`--audio-field` and `--text-field`. The `external` alias requires the precise
published split/config used for the paper; this was not recoverable from the
checked-in scripts and remains a reproducibility TODO.

As checked on 2026-08-22, the yoyo Community repository advertises only a
`train` split and its Hugging Face viewer reports an upstream class-label error.
The training configs therefore use deterministic 90/10 Hugging Face slices so
the commands have defined train/evaluation inputs, while clearly marking the
original validation utterance IDs as unrecoverable. Exact paper reproduction
requires the maintainers to publish/fix those split definitions.

GoldSet leakage is blocked in both training entry points. It is intended only
for held-out benchmark evaluation.
