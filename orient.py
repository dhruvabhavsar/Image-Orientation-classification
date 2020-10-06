#!/usr/local/bin/python3
#
# Authors: Dhruva Bhavsar(dbhavsar), Hely Modi(helymodi), Aneri Shah(annishah)
#

import numpy as np
import sys
import pickle
import NeuralNetwork as nn
import knn as knn
import decisiontree as dt

def read_file(filename):
    traindata, trainlabel, trainnames = [], [], []
    with open(filename, 'r') as f:     
        for line in f.read().lower().rstrip().split("\n"):
            a = line.split(" ")
            trainlabel.append(int(a[1]))
            traindata.append(a[2:])
            trainnames.append(a[0])

    trainlabel = np.array(list(trainlabel), 'int_')
    trainlabel = np.reshape(trainlabel, (-1, 1))
    traindata = np.array(list(traindata), 'int_')
    
    return traindata, trainlabel, trainnames

if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise Exception("Usage: ./orient.py test test_file.txt model_file.txt [model]")

    (action, input_file, model_file, model) = sys.argv[1:]

    if model == 'nearest':
        if action == 'train':
            knn.train(input_file, model_file)
        elif action == 'test':
            knn.test(model_file, input_file, 'nearest_output.txt')
        else:
            raise Exception("Usage: ./orient.py test test_file.txt model_file.txt [model]")
    
    elif model == "nnet":
        num_hid_layer = 12
        epochs = 5000
        learning_rate = 0.00004

        if action == 'train':
            data, labels, names = read_file(input_file)
            X, y = data, labels
            # Train the model
            nnet = nn.NeuralNetwork(epochs, learning_rate, X, y, num_hid_layer, model_file)
            ypred, h1, h2, b1, b2 = nnet.fit(X, y)
        
        elif action == 'test':
            # Test the model
            h1, h2, b1, b2 = pickle.load(open(model_file,"rb"))
            data, labels, names = read_file(input_file)
            X, y = data, labels
            
            nnet = nn.NeuralNetwork(epochs, learning_rate, X, y, num_hid_layer, model_file)
            ypred, output = nnet.evaluate(X, y, h1, h2, b1, b2)
        
            # Write to output file
            result = []
            for i in range(len(names)):
                result.append((names[i], output[i]))
            np.savetxt('nnet_output.txt', result, delimiter=' ', fmt="%s")

    elif model == 'tree':
        if action == 'train':
            params = dt.train(input_file, 500, 0.05)
            pickle.dump(params, open(model_file, "wb"))
        elif action == 'test':
            dt.predict(input_file, model_file)
        else:
            raise Exception("Usage: ./orient.py test test_file.txt model_file.txt [model]")

    elif model == 'best':
        if action == 'train':
            knn.train(input_file, model_file)
        elif action == 'test':
            knn.test(model_file, input_file, 'best_output.txt')
        else:
            raise Exception("Usage: ./orient.py test test_file.txt model_file.txt [model]")
    else:
        print('Invalid Model !')
