import matplotlib.pyplot as plt
import torch

def tracking_plot(history):
    fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(12, 6))

    epochs = history["epochs"]
    train_acc = [acc.cpu().item() for acc in history["train_acc"]]
    train_loss = history["train_loss"]
    test_acc = [x.cpu().item() for x in history["test_acc"]]
    test_loss = [x.cpu().item() for x in history["test_loss"]]
    precision = [x for x in history["precision"]]
    recall = [x for x in history["recall"]]
    f1 = [x for x in history["f1"]]


    ax1.plot(epochs, train_acc, label="Train Accuracy", c="orange")
    ax1.plot(epochs, test_acc, label="Test Accuracy", c="green")
    ax1.set(title="Model Accuracy", xlabel="Epochs", ylabel="Accuracy")
    ax1.grid(True)
    ax1.legend(loc="best")

    ax2.plot(epochs, train_loss, label="Train Loss", c="orange")
    ax2.plot(epochs, test_loss, label="Test Loss", c="green")
    ax2.set(title="Model Loss", xlabel="Epochs", ylabel="Loss")
    ax2.grid(True)
    ax2.legend(loc="best")

    ax3.plot(epochs,precision,label="Precision",c="blue",marker="o")
    ax3.plot(epochs,recall,label="Recall",c="orange",marker="s")
    ax3.plot(epochs,f1,label="F1_Score",c="green",marker="^")
    ax3.set(title="Evaluation Metrics Score",xlabel="Epochs",ylabel="Score")
    ax3.grid(True)
    ax3.legend(loc="best")

    plt.tight_layout()
    plt.show()
