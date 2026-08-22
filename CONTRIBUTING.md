# Contributing

Thank you for helping improve South Azerbaijani speech technology. Please open
an issue before changing dataset partitions, normalization, metric definitions,
or published benchmark values. Those are part of the scientific protocol.

For code changes:

1. Create a focused branch.
2. Install with `pip install -e '.[test]'`.
3. Add or update lightweight tests; tests must not download ASR checkpoints.
4. Run `pytest` and `python -m compileall -q src scripts tests`.
5. Explain any effect on reproducibility in the pull request.

Do not commit recordings, checkpoints, credentials, personal data, or local
filesystem paths. Dataset contributions must have documented consent and a
license compatible with their intended public use.
