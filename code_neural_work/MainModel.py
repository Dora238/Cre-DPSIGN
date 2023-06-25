import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    """
    Inputs                                 Linear/Function                               Output
    [batch_size, 1, 28, 28]          -> Linear(28*28, hidden_size1)          -> [batch_size, hidden_size1]  # first hidden layer
                                     -> Tanh                                 -> [batch_size, hidden_size1]  # tanh activation function, may sigmoid
                                     -> Linear(hidden_size1, hidden_size2)   -> [batch_size, hidden_size2]  # second hidden layer
                                     -> Tanh                                 -> [batch_size, hidden_size2]  # tanh activation function, may sigmoid
                                     -> Linear(hidden_size2, 10)             -> [batch_size, 10]  # Classification Layer
   """

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.classification_layer = nn.Linear(hidden_size2, output_size)
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()

    def forward(self, x):
        """
        :param x: [batch_size, channel, height, width], input for network
        :return: [batch_size, n_classes], output from network
        """
        out = x.view(x.size(0), -1)  # flatten x in [batch_size, dimensions]
        out = self.hidden1(out)
        out = self.tanh1(out)
        out = self.hidden2(out)
        out = self.tanh2(out)
        out = self.classification_layer(out)
        out = F.log_softmax(out, dim=1)
        return out


def flatten_list(message):
    """
    Flatten matrix into vector

    :param message: the gradients from the workers, type:matrix
    """
    wList = [torch.cat([p.flatten() for p in parameters]) for parameters in message]
    wList = torch.stack(wList)
    return wList


def unflatten_vector(vector, model):
    """
    Unflatten vector into matrix which has same size as model parameters

    :param vector: the flattened gradients from the workers, type:vector
    :param model: the global model
    :return:
    """
    paraGroup = []
    cum = 0
    for p in model.parameters():
        newP = vector[cum:cum+p.numel()]
        paraGroup.append(newP.view_as(p))
        cum += p.numel()
    return paraGroup


def caL_loss_acc(model, device, data_loader):
    """
    Calculate loss and accuracy

    :param model: global model
    :param device: gpu or cpu
    :param data_loader: the loader of training data or testing data
    """
    # model.eval()
    loss = 0
    correct = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    return loss, acc


def mean(worker_grad):
    """
    Return the mean of workers' gradients
    """
    return torch.mean(worker_grad, dim=0)


def gm(worker_grad):
    """
    Return the geometric median of workers' gradients
    """
    max_iter = 80
    tol = 1e-5
    guess = torch.mean(worker_grad, dim=0)
    for _ in range(max_iter):
        dist_li = torch.norm(worker_grad - guess, dim=1)
        for i in range(len(dist_li)):
            if dist_li[i] == 0:
                dist_li[i] = 1
        temp1 = torch.sum(torch.stack([w/d for w, d in zip(worker_grad, dist_li)]), dim=0)
        temp2 = torch.sum(1/dist_li)
        guess_next = temp1 / temp2
        guess_movement = torch.norm(guess - guess_next)
        guess = guess_next
        if guess_movement <= tol:
            break
    return guess