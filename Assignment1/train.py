import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torchvision.utils import save_image

from Assignment1.autoencoder import Autoencoder

# print(torch.cuda.is_available())

def init_weights(m, mean=0, std=0.3):
    """
    Initializes the weights of linear layers of m using random normal distribution with mean=0 and variance = 0.2
    :param m:
    :return:
    """

    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean, std)
        torch.nn.init.normal_(m.bias, mean, std)


def train(dataloader, iters = 20):
    """
    Trains the 'model' (autoencoder) on the given dataloader and saves the last batch of images
    :param dataloader: The autoencoder is trained on the dataloader
    :param iters: iterations for training
    :return:
    """
    for itr in range(iters):
        av_itr_loss = 0.0
        for batch_id, (x, _) in enumerate(dataloader):
            optimizer.zero_grad()
            x = x.cuda()
            # x = (x>0.5).float() * 1
            x = x.view(batch_size, -1)
            # print(x[0])
            z = model(x)
            # print((z==1).sum())
            batch_loss = loss(z, x)
            batch_loss.backward()
            optimizer.step()
            av_itr_loss += (1/batch_size)*batch_loss.item()
        save_image(z.view(batch_size, 1, 28, 28), "./{}_Images/generated_image_itr_{}.png".format(loss_type, itr))
        if itr == iters-1:
            save_image(z.view(batch_size,1,28,28),"./{}_generated_image_itr_{}.png".format(loss_type,itr))
        print("Epoch {}: Loss={}".format(itr, av_itr_loss))
    return


if __name__ == "__main__":
    # The networks are initialized with weights sampled from a normal distribution with mean = 0 and std = 0.2.
    # Here only 1 autoencoder (784-256-128-256-784) with sigmoid units and MSE and BCE Losses are tested on the
    # MNIST dataset for 20 iterations. Vanilla SGD was used in both cases to preserve uniformity for comparison
    # as much as possible

    lr = 0.01
    batch_size = 64

    d_train = datasets.MNIST('./data/mnist', train=True, download=True, transform=transforms.ToTensor())
    # d_test  = datasets.MNIST('./data/mnist', train=False, download=True, transform=transforms.ToTensor())

    train_dataloader = torch.utils.data.DataLoader(d_train, batch_size=batch_size, shuffle=True, drop_last=True)
    # test_dataloader  = torch.utils.data.DataLoader(d_test,  batch_size=batch_size, shuffle=True, drop_last=True)

    # Creating Autoencoder of as specified in Question 1
    model = Autoencoder(784, [256, 128], 'sigmoid')

    # Initializing autoencoder weights as specified in Question 2
    model.apply(init_weights)
    model = model.cuda()
    model.train()

    print("Training on MSE Loss:")
    # Using Squared Error Loss as in Question 3
    loss_type = "Squared_Error"
    loss = lambda x, z: (x-z).pow(2).sum(dim=1).mean() # Manual MSELoss implementation with mean over batch of sum(X-Z)^2
    # loss = nn.MSELoss() # MSE either does mean of everything or sum of everything I think
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train(train_dataloader)

    print("\nTraining on Cross Entropy Loss:")
    # Using Cross Entropy Error as in Question 4
    lr = 0.5 # Need a higher lr for CE to converge
    loss_type = "Cross_Entropy"
    model.apply(init_weights) # Reinitializing weights
    model.train()
    # loss = lambda x, z: (x*torch.log1p(z) + (1-x) * torch.log1p(1-z)).sum(dim=1).mean()
    loss = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train(train_dataloader)
