import torch
import torchaudio
import torchaudio.transforms as T

from src.config import SAMPLING_RATE, N_MELS, TOP_DB, DEVICE, BEST_WEIGHTS_PATH, TOKENIZER_NAME
from src.models.deepspeech2 import DeepSpeech2


def load_model(weights_path=BEST_WEIGHTS_PATH, vocab_size=None, device=DEVICE):
    from transformers import Wav2Vec2CTCTokenizer
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_NAME)
    if vocab_size is None:
        vocab_size = tokenizer.vocab_size

    model = DeepSpeech2(vocab_size=vocab_size)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model, tokenizer


def inference(audio_array, orig_sr, model, tokenizer, sampling_rate=SAMPLING_RATE, device=DEVICE):
    audio2mels = T.MelSpectrogram(sample_rate=sampling_rate, n_mels=N_MELS)
    amp2db = T.AmplitudeToDB(top_db=TOP_DB)

    audio = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)

    if orig_sr != sampling_rate:
        audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=sampling_rate)

    mel = audio2mels(audio)
    mel = amp2db(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    mel = mel.unsqueeze(0)
    src_len = torch.tensor([mel.shape[-1]])

    model = model.to(device)

    with torch.no_grad():
        pred_logits, _ = model(mel.to(device), src_len)

    pred_tokens = pred_logits.squeeze().argmax(axis=-1).tolist()
    pred_transcript = tokenizer.decode(pred_tokens)

    return pred_transcript
