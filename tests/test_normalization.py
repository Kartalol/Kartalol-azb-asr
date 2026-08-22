from kartalol_azb_asr.normalization import normalize_azb, number_to_azb


def test_published_normalization_preserves_azb_letters() -> None:
    text = "  بۇ، (آچیقلاما) «سیلینسین» ۱۲۳!  "
    assert normalize_azb(text) == "بۇ، یۆز ایگیرمی اۆچ"


def test_canonical_arabic_persian_variants() -> None:
    assert normalize_azb("كتاب يازى\u200f", profile="canonical") == "کتاب یازی"


def test_number_lexicon_matches_recovered_cleaner() -> None:
    assert number_to_azb(2048) == "ایکی مین قؽرخ سککیز"
    assert normalize_azb("سال ۲۰۲۶") == "سال ایکی مین ایگیرمی آلتی"
