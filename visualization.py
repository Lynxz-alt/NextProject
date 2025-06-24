import matplotlib.pyplot as plt

def plot_loss(history, optimizer_name):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], marker='o')
    plt.title(f'Training Loss per Epoch - {optimizer_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'loss_{optimizer_name}.png')
    plt.show()