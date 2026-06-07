import torch


def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x["input_values"].shape[0], reverse=True)

    batch_mels = [sample["input_values"] for sample in batch]
    batch_transcripts = [sample["labels"] for sample in batch]

    seq_lens = torch.tensor([b.shape[0] for b in batch_mels], dtype=torch.long)

    spectrograms = torch.nn.utils.rnn.pad_sequence(batch_mels, batch_first=True, padding_value=0)

    spectrograms = spectrograms.unsqueeze(1).transpose(-1, -2)

    target_lengths = torch.tensor([len(t) for t in batch_transcripts], dtype=torch.long)

    packed_transcripts = torch.cat(batch_transcripts)

    return {
        "input_values": spectrograms,
        "seq_lens": seq_lens,
        "labels": packed_transcripts,
        "target_lengths": target_lengths,
    }
