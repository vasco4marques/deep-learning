#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        y_pred = np.argmax(np.dot(self.W, x_i)) # Dot product entre input e weights para calcular o y_pred

        if y_pred != y_i: # Errou
            self.W[y_i] +=  x_i #Pode-se adicionar learning rate manualmente mas não fez grande diferença
            self.W[y_pred] -=  x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        label_scores = np.expand_dims(np.dot(self.W,x_i), axis = 1) 

        y_one_hot = np.zeros((np.size(self.W, 0),1))
        y_one_hot[y_i] = 1

        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))

        gradient = (y_one_hot - label_probabilities).dot(np.expand_dims(x_i, axis = 1).T) 

        if l2_penalty > 0:
            gradient += l2_penalty * (np.sum(self.W**2))

        self.W = self.W + (learning_rate * gradient) 


class MLP(object):

    #Hidden layer activation - ReLU
    #Output layer activation - Cross-entropy
    
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.hidden_size= hidden_size
        self.n_classes = n_classes #outputs logo 1 node por classe
        self.n_features = n_features # inputs
        self.weights = []
        self.bias = []

        #Initialize weights and biases for initial layer (initial -> hidden all weights and biases covered)
        self.bias.append(np.zeros((hidden_size, 1)))
        self.weights.append(np.random.normal(0.1,0.1,(hidden_size, n_features)))

        #Initialize weights and biases for hidden layer (hidden -> output all weights and biases covered)
        self.bias.append(np.zeros((n_classes, 1)))
        self.weights.append(np.random.normal(0.1,0.1,(n_classes, hidden_size)))
        
    def relu(self, x):    
        return max(0,x)

    def reluNP(self, x):
        func = np.vectorize(self.relu)
        return func(x)

    def reluDerivative(x):
        for i in range(len(x)):
            x[i] = (x[i] > 0).astype(float)
        return x

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def crossEntropy(self, y_true, y_hat):
        # y - true label index
        # y_hat - predicted values
        
        # Acho que isto (-np.log(y_hat[y])) não chega, ela está simplificada só para o caso de y_hat ser a label certa mas pode não ser
        # logo falta o resto 


        return -np.log(y_hat[y_true])

    def lossGradientAtOutputPreActivation(self, y_hat, y_true):
        y_one_hot = np.zeros((np.size(self.W, 0),1))
        y_one_hot[y_true] = 1


        return y_hat - y_one_hot

    
        
    def forward(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes.
        print(np.shape(X))
        print(np.shape(self.weights[0]))
        print(np.shape(self.weights[1]))
        print(np.shape(self.bias[0]))
        print(np.shape(self.bias[1]))
        z_hidden = np.dot(self.weights[0], np.transpose(X)) + self.bias[0]
        print(np.shape(z_hidden))
        y_hidden = self.reluNP(z_hidden)
        z_hat = np.dot(self.weights[1], y_hidden) + self.bias[1]    
        y_hat = self.softmax(z_hat)

        print(np.shape(z_hidden))
        print(np.shape(y_hidden))
        print(np.shape(z_hat))
        print(np.shape(y_hat))

        return z_hidden, y_hidden, z_hat, y_hat
    
    
    def predict(self, X):
        # Predicts the label using the output of the NN 

        _,_,_,y_hat = self.forward(X)
    
        return np.argmax(y_hat, axis = 0)
        

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        print(np.shape(y_hat))
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def backPropagate(self, y_hat, z_hat, y_hidden, z_hidden, x, y_true):
        outputGradient = self.lossGradientAtOutputPreActivation(y_hat, y_true)
        weightGradient = self.weights
        biasGradient = self.bias

        weightGradient[1] = np.dot(outputGradient, np.transpose(y_hidden))
        biasGradient[1] = outputGradient

        hDerivative = np.dot(self.weights[1], outputGradient)

        outputGradient = np.multiply(hDerivative, self.reluDerivative(z_hidden))

        weightGradient[0] = np.dot(outputGradient, np.transpose(x))
        biasGradient[0] = outputGradient



    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        """
        Dont forget to return the loss of the epoch.
        """
        losses = []
        X = np.transpose(X)
        for i in range(len(X)):
            z_hidden, y_hidden, z_hat, y_hat = self.forward(X[i])

            losses[i] = self.crossEntropy(y, y_hat)

            outputGradient = self.lossGradientAtOutputPreActivation(y_hat, y)
            weightGradient = self.weights
            biasGradient = self.bias

            weightGradient[1] = np.dot(outputGradient, np.transpose(y_hidden))
            biasGradient[1] = outputGradient

            hDerivative = np.dot(self.weights[1], outputGradient)

            outputGradient = np.multiply(hDerivative, self.reluDerivative(z_hidden))

            weightGradient[0] = np.dot(outputGradient, np.transpose(X[i]))
            biasGradient[0] = outputGradient

            self.weights[1] += weightGradient[1]*learning_rate
            self.weights[0] += weightGradient[0]*learning_rate
            self.bias[1] += biasGradient[1]*learning_rate
            self.bias[0] += biasGradient[0]*learning_rate

        return np.sum(losses)/len(losses)


def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='intel_landscapes.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    with open(f"Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")


if __name__ == '__main__':
    main()
