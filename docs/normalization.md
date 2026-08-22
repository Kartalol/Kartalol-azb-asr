# South Azerbaijani normalization

Official benchmark evaluation is raw (`--normalization none`). Normalized
scores are an additional analysis and must be requested explicitly.

The `published` profile preserves the recoverable behavior of the original
paper-era cleaner: Arabic/Persian digits are mapped to ASCII, integers up to
four digits are expanded with the original South Azerbaijani number lexicon,
parenthesized and guillemet-delimited material is removed, ASCII punctuation is
deleted, and whitespace is collapsed.

The `canonical` profile first applies Unicode NFKC, removes bidi formatting
controls, and maps Arabic yeh/kaf variants to their Persian forms. It is useful
for analysis but did not appear in the original training scripts, so it must not
be presented as an official published score without rerunning the benchmark.
