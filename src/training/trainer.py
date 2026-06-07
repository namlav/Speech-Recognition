import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.config import (
    BATCH_SIZE, TRAINING_ITERATIONS, EVAL_ITERATIONS, LEARNING_RATE,
    NUM_WORKERS, DEVICE, BEST_WEIGHTS_PATH,
)
from src.data import collate_fn
from src.models.deepspeech2 import DeepSpeech2
from datasets import load_dataset


def train():
    from src.data.dataset import LSVSCDataset
    from src.config import TOKENIZER_NAME
    from transformers import Wav2Vec2CTCTokenizer

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_NAME)
    lsvsc_dataset = load_dataset("doof-ferb/LSVSC")

    train_hf = lsvsc_dataset["train"]
    test_hf = lsvsc_dataset["validation"] if "validation" in lsvsc_dataset else lsvsc_dataset["test"]

    trainset = LSVSCDataset(hf_dataset=train_hf, tokenizer=tokenizer, is_train=True)
    testset = LSVSCDataset(hf_dataset=test_hf, tokenizer=tokenizer, is_train=False)

    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True,
    )
    testloader = DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True,
        drop_last=True,
    )

    model = DeepSpeech2(vocab_size=tokenizer.vocab_size).to(DEVICE)
    params = sum([p.numel() for p in model.parameters()])
    print("Total Training Parameters:", params)

    optimizer = optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=TRAINING_ITERATIONS,
    )

    best_val_loss = np.inf
    completed_steps = 0
    train_loss, validation_loss = [], []
    pbar = tqdm(range(TRAINING_ITERATIONS))
    scaler = torch.amp.GradScaler("cuda")

    while completed_steps < TRAINING_ITERATIONS:
        training_losses = []
        validation_losses = []

        model.train()
        for batch in trainloader:
            with torch.amp.autocast("cuda"):
                logits, input_lengths = model(
                    x=batch["input_values"].to(DEVICE), seq_lens=batch["seq_lens"],
                )

                log_probs = nn.functional.log_softmax(logits, dim=-1)
                log_probs = log_probs.transpose(0, 1)

                loss = nn.functional.ctc_loss(
                    log_probs=log_probs,
                    targets=batch["labels"].to(DEVICE),
                    input_lengths=input_lengths,
                    target_lengths=batch["target_lengths"],
                    blank=tokenizer.pad_token_id,
                    reduction="mean",
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            training_losses.append(loss.item())

            completed_steps += 1
            pbar.update(1)

            if completed_steps % EVAL_ITERATIONS == 0:
                print("Evaluating")

                model.eval()
                for batch in tqdm(testloader):
                    with torch.no_grad():
                        logits, input_lengths = model(
                            x=batch["input_values"].to(DEVICE), seq_lens=batch["seq_lens"],
                        )

                    log_probs = nn.functional.log_softmax(logits, dim=-1)
                    log_probs = log_probs.transpose(0, 1)

                    loss = nn.functional.ctc_loss(
                        log_probs=log_probs,
                        targets=batch["labels"].to(DEVICE),
                        input_lengths=input_lengths,
                        target_lengths=batch["target_lengths"],
                        blank=tokenizer.pad_token_id,
                        reduction="mean",
                        zero_infinity=True,
                    )

                    validation_losses.append(loss.item())

                training_loss_mean = np.mean(training_losses)
                valid_loss_mean = np.mean(validation_losses)

                train_loss.append(training_loss_mean)
                validation_loss.append(valid_loss_mean)

                if valid_loss_mean < best_val_loss:
                    print("---Saving Model---")
                    torch.save(model.state_dict(), BEST_WEIGHTS_PATH)
                    best_val_loss = valid_loss_mean

                print("Training Loss:", training_loss_mean)
                print("Validation Loss:", valid_loss_mean)

                training_losses = []
                validation_losses = []
                model.train()

            if completed_steps >= TRAINING_ITERATIONS:
                print("Completed Training")
                break

    return train_loss, validation_loss
