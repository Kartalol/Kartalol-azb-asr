# Historical experiments

The pre-cleanup repository contained exploratory, machine-specific scripts for:

- duplicate Whisper Tiny/Base/Small fine-tuning;
- `facebook/wav2vec2-base` CTC training with a generated character vocabulary;
- MMS-1B-All AZB adapter training;
- Wav2Vec2-BERT + KenLM decoding and Optuna tuning;
- Telegram-based voice collection and local dataset conversion.

Reusable Whisper and MMS behavior now lives in `scripts/` and `src/`. The old
files contained fixed developer paths, notebook outputs, breakpoints, local
dataset assumptions, and—within an untracked prototype—an exposed Hugging Face
token. They remain recoverable from Git history but are intentionally not part
of the runnable public tree.

`kenlm_best_params.json` records the historical Optuna result. It is not used by
the INTERSPEECH 2026 benchmark CLI and must not be interpreted as a parameter
for the reported Whisper/MMS experiments.
