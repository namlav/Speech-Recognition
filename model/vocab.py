BLANK = '<blank>'
SPACE = ' '

_chars = list('abcdefghijklmnopqrstuvwxyzđ')
_vowels = list('áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ')

VOCAB = [BLANK, SPACE] + _chars + _vowels

char_to_int = {c: i for i, c in enumerate(VOCAB)}
int_to_char = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
BLANK_IDX = 0


def text_to_int(text):
    return [char_to_int.get(c, char_to_int[SPACE]) for c in text.lower()]


def int_to_text(ints):
    return ''.join(int_to_char[i] for i in ints)
