import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F

from src.config import SAMPLING_RATE, N_MELS, TOP_DB, SPECAUG_FREQ_MASK_PARAM, SPECAUG_TIME_MASK_PARAM


class LSVSCDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, sampling_rate=SAMPLING_RATE, is_train=True):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.sampling_rate = sampling_rate
        self.is_train = is_train

        self.audio2mels = T.MelSpectrogram(
            sample_rate=sampling_rate,
            n_mels=N_MELS
        )
        self.amp2db = T.AmplitudeToDB(top_db=TOP_DB)

        if self.is_train:
            self.freq_masking = T.FrequencyMasking(freq_mask_param=SPECAUG_FREQ_MASK_PARAM)
            self.time_masking = T.TimeMasking(time_mask_param=SPECAUG_TIME_MASK_PARAM)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        audio_array = item["audio"]["array"]
        orig_sr = item["audio"]["sampling_rate"]

        audio = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)

        if orig_sr != self.sampling_rate:
            audio = F.resample(audio, orig_freq=orig_sr, new_freq=self.sampling_rate)

        mel = self.audio2mels(audio)
        mel = self.amp2db(mel)

        if self.is_train:
            mel = self.freq_masking(mel)
            mel = self.time_masking(mel)

        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        transcript = item["transcription"].lower()
        tokenized_transcript = torch.tensor(self.tokenizer.encode(transcript))

        return {
            "input_values": mel[0].T,
            "labels": tokenized_transcript,
        }
