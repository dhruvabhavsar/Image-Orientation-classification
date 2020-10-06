import numpy as np
import pickle

def sigmoid(X, derivative=False):
    if derivative:
        return np.multiply(X, (1.0 - X))
    return (1 / (1 + np.exp(-X)))

def calc_accuracy(pred_y, actual_y):
    to_file = []
    count = 0
    for i in range(len(pred_y)):
        t = -1
        max = np.argmax(pred_y[i])
        if max == 0:
            t = 0
        elif max == 1:
            t = 90
        elif max == 2:
            t = 180
        elif max == 3:
            t = 270

        if t == actual_y[i][0]:
            count += 1
        to_file.append(t)

    accuracy = count/len(actual_y) * 100
    return accuracy, to_file

class NeuralNetwork(object):
    def __init__(self, epochs, learning_rate, X, y, hidden_layer, model_file):
        self.model_file = model_file
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.y = self.initialize_output(y)

        # Initialize the weights
        self.h1 = np.asmatrix(np.random.rand(X.shape[1],hidden_layer))-np.asmatrix(np.random.rand(X.shape[1],hidden_layer))
        self.h2 = np.asmatrix(np.random.rand(hidden_layer, self.y.shape[1]))-np.asmatrix(np.random.rand(hidden_layer, self.y.shape[1]))
        self.b1 = np.asmatrix(np.random.rand(1,hidden_layer))-np.asmatrix(np.random.rand(1,hidden_layer))
        self.b2 = np.asmatrix(np.random.rand(1,self.y.shape[1]))-np.asmatrix(np.random.rand(1,self.y.shape[1]))

    def initialize_output(self, y):
        z = []
        for i in y:
            if (i == 0):
                z.append([1, 0, 0, 0])
            elif (i == 90):
                z.append([0, 1, 0, 0])
            elif (i == 180):
                z.append([0, 0, 1, 0])
            elif (i == 270):
                z.append([0, 0, 0, 1])
        return np.asarray(z)

    def forward(self, X, y):
        self.z1 = np.dot(X, self.h1) + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.h2) + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    def backward(self, X, y):
        
        # Calculating error of each layer
        d3 = self.y - self.a2

        delta2 = np.multiply(d3, sigmoid(self.a2, True))
        delta1 = np.multiply(np.dot(delta2, np.transpose(self.h2)), sigmoid(self.a1, True))

        # Updating the weights using gradient descent
        self.h2 += self.learning_rate * np.dot(np.transpose(self.a1), delta2)
        self.b2 += self.learning_rate * np.sum(delta2)
        self.h1 += self.learning_rate * np.dot(np.transpose(X), delta1)
        self.b1 += self.learning_rate * np.sum(delta1)

    def fit(self, X, y):
        o = np.zeros((X.shape[0], 4))
        for i in range(self.epochs):
            o = self.forward(X, y)
            self.backward(X, y)

        #save trained weights
        pickle.dump([self.h1, self.h2, self.b1, self.b2], open(self.model_file, "wb"))
        return o, self.h1, self.h2, self.b1, self.b2

    def evaluate(self, X, y, w1, w2, b1, b2):
        self.h1 = w1
        self.h2 = w2
        self.b1 = b1
        self.b2 = b2
        o = self.forward(X, y)

        acc, output = calc_accuracy(o, y)
        print("Test Accuracy:", acc, "%")

        return o, output
