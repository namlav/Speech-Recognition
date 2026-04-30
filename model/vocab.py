BLANK = '_'
SPACE = ' '
APOSTROPHE = "'"

# Standard DeepSpeech2 29-token English vocabulary (matching the checkpoint)
# Blank must be at index 0 for CTC loss
VOCAB = [
    BLANK,        # 0
    APOSTROPHE,   # 1
    SPACE,        # 2
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
]

char_to_int = {c: i for i, c in enumerate(VOCAB)}
int_to_char = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
BLANK_IDX = 0


def text_to_int(text):
    return [char_to_int.get(c, char_to_int[SPACE]) for c in text.lower()]


def int_to_text(ints):
    return ''.join(int_to_char[i] for i in ints)
