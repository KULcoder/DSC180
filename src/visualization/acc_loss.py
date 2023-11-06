
import matplotlib.pyplot as plt

"""
Functions:
    plot_acc_loss
"""

def plot_acc_loss(training_losses, training_accs, valid_losses, valid_accs, path=None):
    """
    Plot training and validation losses and accuracies.
    If path is provided, save figure to path.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(training_losses, label='Training')
    ax1.plot(valid_losses, label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(training_accs, label='Training')
    ax2.plot(valid_accs, label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    if path:
        plt.savefig(path)
    plt.show()