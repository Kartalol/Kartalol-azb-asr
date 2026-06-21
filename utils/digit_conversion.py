import re

# Define dictionary mapping for numbers
_num_words = {
    1: "بیر",
    2: "ایکی",
    3: "اۆچ",
    4: "دؤرد",
    5: "بئش",
    6: "آلتی",
    7: "یئددی",
    8: "سککیز",
    9: "دوْققوز",
    10: "اوْن",
    20: "ایگیرمی",
    30: "اوْتوز",
    40: "قؽرخ",
    50: "اللی",
    60: "آلتمیش",
    70: "یئتمیش",
    80: "هشتاد",
    90: "دوْخسان",
    100: "یۆز",
    1000: "مین",
}

def number_to_words(num):
    """Convert a number (1-9999) into Arabic words."""
    if num == 0:
        return "صفر"

    words = []

    # Handle thousands (1000-9999)
    if num >= 1000:
        thousands = num // 1000
        if thousands > 1:
            words.append(_num_words[thousands] + " " + _num_words[1000])
        else:
            words.append(_num_words[1000])
        num %= 1000  # Reduce number

    # Handle hundreds (100-999)
    if num >= 100:
        hundreds = num // 100
        if hundreds > 1:
            words.append(_num_words[hundreds] + " " + _num_words[100])
        else:
            words.append(_num_words[100])
        num %= 100  # Reduce number

    # Handle tens (20, 30, ..., 90)
    if num >= 10:
        tens = (num // 10) * 10
        words.append(_num_words[tens])
        num %= 10  # Reduce number

    # Handle 1-9
    if num > 0:
        words.append(_num_words[num])

    return " ".join(words)

def replace_numbers_in_text(text):
    """Find numbers (1-9999) in Arabic sentences and replace them with words."""
    def replace_match(match):
        num = int(match.group())  # Extract number
        return number_to_words(num)  # Convert to words

    return re.sub(r"\b\d{1,4}\b", replace_match, text)
"""
# Example Usage
arabic_text = "12 اون ایکی و 345 یۆز آلتی دؤرد و 6789 مین"
converted_text = replace_numbers_in_text(arabic_text)
print(converted_text)
"""