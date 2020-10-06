## A4: Machine Learning

### Neural Networks
1. Problem Formulation
- For this learning model, we used the numpy library to handle the matrix multiplication efficiently and effectively.
- The algorithm of Gradient Descent has been used for backpropagation.


2. Description
- This problem has 192 features, 12 neurons in the hidden layer and 4 neurons in the output layer(one for each image orientation).
- The learning rate has been fixed at 0.00004 and the model learns for around 5000 iterations.
- The sigmoid activation function is used for all the layers.
- The output is being stored to the output file named nnet_output.txt.


3. Discussion
- We tried various different combinations for the classifier, i.e the learning rate, number of hidden neurons, activation functions, normalization of the data etc.
- For the testing phase, we have used the weights stored from the training phase and used that to calculate the test accuracy. The output is being stored in a file called nnet_model.txt.
- Initially we began with 4 neurons in the hidden layer with learning rate of 0.01 and found that the model achieved around 25% accuracy. We then increased the number of neurons, however, the accuracy did not increase much. We tried again using neurons between the range 5-10 with a learning rate of 0.00005 and found our best accuracy at 72.26% using 12 hidden neurons with learning rate as 0.00004. The table below presents the different combinations performed.

Input:
```
./orient.py train train_file.txt nnet_model.txt nnet
./orient.py test test_file.txt nnet_model.txt nnet
```

| Hidden Neurons         | Learning Rate     | Test Accuracy  |
| ------------- |:-------------:| -----:|
| 4      | 0.01 | 25.8% |
| 4      | 0.001      |   42.33% |
| 5 | 0.00001      |    69% |
| 10 | 0.00005     |    70.47% |
| 12 | 0.00004      |    72.26% |



### Decision Trees

1. Problem Formulation:-
- First of all to learn this model, we calculated the entropy to decide on the best (feature, split) pair. 
- We have used ID3 algorithm for learning this model and also used numpy library for handling mathematical computations efficiently and effectively.

2. Description:-
- The model is trained by creating small  subset of trees which  learns the features and then after generating a combined tree.
- This combined tree expands the nodes in accordance with the minimum entropy value and prepares the training model.
- The depth of the tree is randomly choosen from range(2,5) and the dataset is learned for around 800 counts.

3. Discussion:-
- For testing the dataset, we have used the splits from the training model and calculated the accuracy of the test set. The output is being stored to the output file named tree_output.txt.
- We have tried different combinations of tree-counts and ratio to obtain best accuracy of which we obtained the best accuracy of 68.98% with 500 tree-counts and ratio of 0.05. The table below shows other combinations:-

Input:
```
./orient.py train train_file.txt tree_model.txt tree
./orient.py test test_file.txt tree_model.txt tree
```

| Tree count   | Ratio    | Test Accuracy  |
| -------------|:--------:| --------------:|
|500           | 0.04     | 67.12%         |
|500           | 0.01     |   65.42%       |
|500           | 0.05     |     68.98%     |

### K Nearest Neighbours

- This model uses numpy to calculate the Euclidean distance between the points in train dataset and test dataset.
- For selecting the value of k, we ran the code for k in range of 5 to 200 with increments of 5. The value of k with maximum accuracy has been used in the final code. The code used for selecting the value of k can be found in knn_graph.py .
- The output is being stored to the output file named nearest_output.txt.
- The graph of value of K vs. the accuracy is as below:

![KNN Graph](/knn_graph.png)

- For k=25, the accuracy obtained is 72.85% and the code takes about 40 seconds to execute.

Input:
```
./orient.py train train_file.txt nearest_model.txt nearest
./orient.py test test_file.txt nearest_model.txt nearest
```

Output:
```
Accuracy: 72.8525980911983 %
```
### Best Model

- In our case, the best accuracy was obtained using k nearest neighbours. Hence, we have used KNN as our best model.
- The best accuracy obtained was 72.85% .
- The output is being stored to the output file named best_output.txt.

Input:
```
./orient.py train train_file.txt best_model.txt best
./orient.py test test_file.txt best_model.txt best
```

Output:
```
Accuracy: 72.8525980911983 %
```

