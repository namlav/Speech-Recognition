import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(train_loss, validation_loss):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss", marker='o')
    plt.plot(epochs, validation_loss, label="Validation Loss", marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()
