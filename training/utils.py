CHO = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]

JUNG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"
]

JONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]


def is_hangul_syllable(ch: str) -> bool:
    return 0xAC00 <= ord(ch) <= 0xD7A3


def decompose_syllable(ch: str):
    code = ord(ch) - 0xAC00
    cho_idx = code // 588
    jung_idx = (code % 588) // 28
    jong_idx = code % 28
    return CHO[cho_idx], JUNG[jung_idx], JONG[jong_idx]


def token_to_jamo_tokens(decoded_token: str):
    chos, jungs, jongs = [], [], []

    for ch in decoded_token:
        if is_hangul_syllable(ch):
            c, v, f = decompose_syllable(ch)
            chos.append(c)
            jungs.append(v)
            jongs.append(f if f else "_")

    if not chos:
        return None, None, None

    return (
        "<CHO:" + "".join(chos) + ">",
        "<JUNG:" + "".join(jungs) + ">",
        "<JONG:" + "".join(jongs) + ">",
    )


def make_jamo_labels(input_ids, base_tokenizer, jamo_tokenizer):
    cho_labels = []
    jung_labels = []
    jong_labels = []

    unk_id = jamo_tokenizer.unk_token_id

    for token_id in input_ids:
        decoded = base_tokenizer.decode(
            [int(token_id)],
            clean_up_tokenization_spaces=False,
        )

        cho_tok, jung_tok, jong_tok = token_to_jamo_tokens(decoded)

        if cho_tok is None:
            cho_labels.append(-100)
            jung_labels.append(-100)
            jong_labels.append(-100)
            continue

        cho_id = jamo_tokenizer.convert_tokens_to_ids(cho_tok)
        jung_id = jamo_tokenizer.convert_tokens_to_ids(jung_tok)
        jong_id = jamo_tokenizer.convert_tokens_to_ids(jong_tok)

        cho_labels.append(-100 if cho_id == unk_id else cho_id)
        jung_labels.append(-100 if jung_id == unk_id else jung_id)
        jong_labels.append(-100 if jong_id == unk_id else jong_id)

    return cho_labels, jung_labels, jong_labels