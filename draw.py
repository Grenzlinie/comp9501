def plot_losses():
    # read file from csv
    from matplotlib import pyplot as plt
    import pandas as pd

    data = pd.read_csv('./Results/loss_results.csv')

    # extract epoch, d loss and g loss
    epochs = data['Epoch']
    d_loss = data['D Loss']

    # plot
    plt.figure()
    plt.plot(epochs, d_loss, label='D Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./Results/plot_loss.jpg')
    plt.show()

if __name__ == '__main__':
    plot_losses()