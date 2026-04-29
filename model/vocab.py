# model/vocab.py
import torch

class CharVocabulary:
    def __init__(self):
        # Ký tự Blank cho CTCLoss bắt buộc ở vị trí 0
        self.chars = ["<BLANK>", " ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", 
                      "á", "à", "ả", "ã", "ạ", "ă", "ắ", "ằ", "ẳ", "ẵ", "ặ", "â", "ấ", "ầ", "ẩ", "ẫ", "ậ",
                      "đ", "é", "è", "ẻ", "ẽ", "ẹ", "ê", "ế", "ề", "ể", "ễ", "ệ",
                      "í", "ì", "ỉ", "ĩ", "ị",
                      "ó", "ò", "ỏ", "õ", "ọ", "ô", "ố", "ồ", "ổ", "ỗ", "ộ", "ơ", "ớ", "ờ", "ở", "ỡ", "ợ",
                      "ú", "ù", "ủ", "ũ", "ụ", "ư", "ứ", "ừ", "ử", "ữ", "ự",
                      "ý", "ỳ", "ỷ", "ỹ", "ỵ"]
        
        self.char_to_id = {char: idx for idx, char in enumerate(self.chars)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def text_to_tensor(self, text):
        text = text.lower()
        # Bỏ qua các ký tự không có trong từ điển
        indices = [self.char_to_id[c] for c in text if c in self.char_to_id]
        return torch.tensor(indices, dtype=torch.long)