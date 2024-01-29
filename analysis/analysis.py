import torch
import matplotlib.pyplot as plt

def plot_train_test():
    x = torch.load("C:\\Work\\EmbryoVision\\data\\torch_type\\train_list")
    y = torch.load("C:\\Work\\EmbryoVision\\data\\torch_type\\test_list")
    plt.plot(x=x,y=y)
    plt.show()