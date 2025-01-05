#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

import time
import utils

# Q
class LogisticRegression(nn.Module):

    def __init__(self, n_classes, n_features, **kwargs):
        """
        n_classes (int)
        n_features (int)

        The __init__ should be used to declare what kind of layers and other
        parameters the module has. For example, a logistic regression module
        has a weight matrix and bias vector. For an idea of how to use
        pytorch to make weights and biases, have a look at
        https://pytorch.org/docs/stable/nn.html
        """
        super().__init__()
        # In a pytorch module, the declarations of layers needs to come after
        # the super __init__ line, otherwise the magic doesn't work.
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples

        Every subclass of nn.Module needs to have a forward() method. forward()
        describes how the module computes the forward pass. In a log-lineear
        model like this, for example, forward() needs to compute the logits
        y = Wx + b, and return y (you don't need to worry about taking the
        softmax of y because nn.CrossEntropyLoss does that for you).

        One nice thing about pytorch is that you only need to define the
        forward pass -- this is enough for it to figure out how to do the
        backward pass.
        """
        return self.linear(x) # Retornamos a self.linear (raw sem ativação visto que o CrossEntropyLoss aplica softmax automaticamente segundo a documentação)


class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        """
        n_classes (int)
        n_features (int)
        hidden_size (int)
        layers (int)
        activation_type (str)
        dropout (float): dropout probability

        As in logistic regression, the __init__ here defines a bunch of
        attributes that each FeedforwardNetwork instance has. Note that nn
        includes modules for several activation functions and dropout as well.
        """
        
        super().__init__()

        activationLayer = None

        if activation_type == 'tanh':
            activationLayer = nn.Tanh()
        elif activation_type == 'relu':
            activationLayer = nn.ReLU()

        layerStructure = []

        if(layers > 0): # Se hidden_size > 0, então temos hidden layers
            layerStructure.append(nn.Linear(n_features, hidden_size))
            layerStructure.append(activationLayer)
            layerStructure.append(nn.Dropout(dropout))

            for i in range(layers-2):
                layerStructure.append(nn.Linear(hidden_size, hidden_size))
                layerStructure.append(activationLayer)
                layerStructure.append(nn.Dropout(dropout))

            layerStructure.append(nn.Linear(hidden_size, n_classes))
        
        else: # Caso contrário, temos apenas a camada de input e output
            layerStructure.append(nn.Linear(n_features, n_classes))

        self.network = nn.Sequential(*layerStructure) # Cria rede sequencial a partir das camadas de layers

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples

        This method needs to perform all the computation needed to compute
        the output logits from x. This will include using various hidden
        layers, pointwise nonlinear functions, and dropout.
        """
        # Não se aplica nada à rede porque o CrossEntropyLoss já aplica softmax
        return self.network(x) 
    


def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function

    To train a batch, the model needs to predict outputs for X, compute the
    loss between these predictions and the "gold" labels y using the criterion,
    and compute the gradient of the loss with respect to the model parameters.

    Check out https://pytorch.org/docs/stable/optim.html for examples of how
    to use an optimizer object to update the parameters.

    This function should return the loss (tip: call loss.item()) to get the
    loss as a numerical value that is not part of the computation graph.
    """

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X)

    # Compute the loss
    sumOfWeights = sum([torch.sum(p ** 2) for p in model.parameters()])
    loss = criterion(outputs, y) + kwargs.get('l2_decay', 0.0) * sumOfWeights #l2 regularization (l2 value * sum dos parametros ao quadrado)

    # Backward pass
    loss.backward()

    # Update the parameters
    optimizer.step()

    return loss.item()



def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


@torch.no_grad()
def evaluate(model, X, y, criterion):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    logits = model(X)
    loss = criterion(logits, y)
    loss = loss.item()
    y_hat = logits.argmax(dim=-1)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return loss, n_correct / n_possible


def plot(epochs, plottables, filename=None, ylim=None):
    """Plot the plottables over the epochs.
    
    Plottables is a dictionary mapping labels to lists of values.
    """
    plt.clf()
    plt.xlabel('Epoch')
    for label, plottable in plottables.items():
        plt.plot(epochs, plottable, label=label)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    if filename:
        plt.savefig(filename, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=200, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=64, type=int,
                        help="Size of training batch.")
    parser.add_argument('-hidden_size', type=int, default=200)
    parser.add_argument('-layers', type=int, default=2)
    parser.add_argument('-learning_rate', type=float, default=0.002)
    parser.add_argument('-l2_decay', type=float, default=0.0)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-momentum', type=float, default=0.0)
    parser.add_argument('-activation',
                        choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-data_path', type=str, default='intel_landscapes.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_dataset(opt.data_path)
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]  # 10
    n_feats = dataset.X.shape[1]

    # initialize the model
    if opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = FeedforwardNetwork(
            n_classes,
            n_feats,
            opt.hidden_size,
            opt.layers,
            opt.activation,
            opt.dropout
        )

    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay, momentum = opt.momentum
    )

    # get a loss criterion
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_losses = []
    valid_losses = []
    valid_accs = []

    start = time.time()

    print('initial val acc: {:.4f}'.format(evaluate(model, dev_X, dev_y, criterion)[1]))

    for ii in epochs:
        print('Training epoch {}'.format(ii))
        epoch_train_losses = []
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            epoch_train_losses.append(loss)

        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

        print('train loss: {:.4f} | val loss: {:.4f} | val acc: {:.4f}'.format(
            epoch_train_loss, val_loss, val_acc
        ))

        train_losses.append(epoch_train_loss)
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    _, test_acc = evaluate(model, test_X, test_y, criterion)
    print('Final test acc: {:.4f}'.format(test_acc))

    # plot
    if opt.model == "logistic_regression":
        config = (
            f"batch-{opt.batch_size}-lr-{opt.learning_rate}-epochs-{opt.epochs}-"
            f"l2-{opt.l2_decay}-opt-{opt.optimizer}"
        )
    else:
        config = (
            f"batch-{opt.batch_size}-lr-{opt.learning_rate}-epochs-{opt.epochs}-"
            f"hidden-{opt.hidden_size}-dropout-{opt.dropout}-l2-{opt.l2_decay}-"
            f"layers-{opt.layers}-act-{opt.activation}-opt-{opt.optimizer}-mom-{opt.momentum}"
        )

    losses = {
        "Train Loss": train_losses,
        "Valid Loss": valid_losses,
    }

    plot(epochs, losses, filename=f'{opt.model}-training-loss-{config}.pdf')
    accuracy = { "Valid Accuracy": valid_accs }
    plot(epochs, accuracy, filename=f'{opt.model}-validation-accuracy-{config}.pdf')


if __name__ == '__main__':
    main()
