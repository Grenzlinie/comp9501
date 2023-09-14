def plot_losses():
    # read file from csv
    from matplotlib import pyplot as plt
    import pandas as pd

    data = pd.read_csv('./Results/results_n.csv')

    # extract epoch, d loss and g loss
    epochs = data['Epoch']
    d_loss = data['D Loss']
    # val_d_loss = data['Val D Loss']
    g_loss = data['G Loss']
    # val_g_loss = data['Val G Loss']
    d_real_loss = data['D Real Loss']
    d_fake_loss = data['D Fake Loss']

    plt.figure()
    plt.plot(epochs, d_loss, label='D Loss')
    plt.plot(epochs, g_loss, label='G Loss')
    plt.plot(epochs, d_real_loss, label='D Real Loss')
    plt.plot(epochs, d_fake_loss, label='D Fake Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./Results/plot_loss.jpg')
    plt.show()

    # # create a new picture
    # plt.figure()

    # # draw g loss
    # plt.plot(epochs, g_loss, label='G Loss')

    # # draw val g loss
    # # plt.plot(epochs, val_g_loss, label='Val G Loss')

    # # add x title
    # plt.xlabel('Epoch')

    # # add y title
    # plt.ylabel('Loss')

    # # add legend
    # plt.legend()

    # # store picture
    # plt.savefig('./Results/plot_g.jpg')

    # # show picture
    # plt.show()

    # # create a new picture
    # plt.figure()

    # # draw d loss
    # plt.plot(epochs, d_loss, label='D Loss')

    # # draw val d loss
    # # plt.plot(epochs, val_d_loss, label='Val D Loss')

    # # add x title
    # plt.xlabel('Epoch')

    # # add y title
    # plt.ylabel('Loss')

    # # add legend
    # plt.legend()

    # # store picture
    # plt.savefig('./Results/plot_d.jpg')

    # # show picture
    # plt.show()

if __name__ == '__main__':
    plot_losses()