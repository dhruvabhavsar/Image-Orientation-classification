import numpy as np
import random
from collections import Counter
import pickle


def loadfile(file):
    input_file = open(file,'rt')
    image_ids = []
    labels = []
    X= []
    for line in input_file:
        words = line.split()
        X.append(list(map(int, words[2:])))
        image_ids.append(words[0])
        labels.append(int(words[1]))
    X = np.array(X)
    return X, np.array(labels), np.array(image_ids)

#Refered from "https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/"
#Finding the best split according to the entropy.
def entropy(X, Y,features):

    splits = np.random.normal(128,30,10).astype(int)
    entropy_dict = {}
    i=0
    Y=np.array(Y)
    for feature in features:
        node = i
        for split in splits:
            
            left_label = Y[np.where(X[node] < split)[0]]
            right_label = Y[np.where(X[node] >= split)[0]]
            left_counts_elements = np.unique(left_label, return_counts=True)[1]
            right_counts_elements = np.unique(right_label, return_counts=True)[1]
            entropy = len(left_label)/len(X)*np.sum(-1 *(left_counts_elements/len(left_label)) * np.log(left_counts_elements/len(left_label))) + \
                        len(right_label)/len(X)*np.sum(-1 *(right_counts_elements/len(right_label)) * np.log(right_counts_elements/len(right_label)))
            entropy_dict[(feature,split)] = entropy
        i+=1                 
    return min(entropy_dict, key = entropy_dict.get)

class Node:
    def __init__(self, data):

        self.left = None
        self.right = None
        self.data = data
        self.depth = 0
        self.pred = 0

def generate_tree(X,Y,depth, parent,features):
    Y=np.array(Y)
    tree = Node(entropy(X,Y,features))
    if parent == None:
        parent = tree
    if parent.depth >= depth:
        return None
    if len(Y)<15 :
        if len(Y)==0: return None 
        a = Counter(Y).most_common(1)
        n= Node(None)
        n.pred = a[0][0]
        return n
    tree.depth = parent.depth+1
    feature_no = parent.data[0]
    root = features.index(feature_no)
    split = parent.data[1]
    X_left_i = np.where(X[root] < split)[0]
    X_left = X[:,X_left_i]
    Y_left = Y[X_left_i]
    X_right_i = np.where(X[root] >= split)[0]
    X_right = X[:,X_right_i]
    Y_right = Y[X_right_i]
    
    tree.left = generate_tree(X_left, Y_left, depth, tree,features)
    tree.right = generate_tree(X_right, Y_right, depth, tree,features)
    if (tree.left==None and tree.right == None):
        a = Counter(Y).most_common(1)
        tree.pred = a[0][0] 
    return tree


def random_generation(X, sub_size,Y=None):
    indices = [random.randint(0,X.shape[0]-1) for i in range(0,sub_size)]
    Xsub = [X[ind] for ind in indices]
    if type(Y)!=type(None):
        Ysub = [Y[ind] for ind in indices]
        return np.array(Xsub),np.array(Ysub)
    return np.array(Xsub),indices

#Function to train the randomly generated tree model.
def train(inputfile, tree_count, ratio):
    X, Y, IDs = loadfile(inputfile)
    X = X.T
    list_trees = []
    for i in range(tree_count):
        subset_size = int(X.shape[1]*ratio)
        Xtmp,features = random_generation(X, 20)                     
        Xsub,Ysub = random_generation(Xtmp.T, subset_size,Y)
        Xsub = np.array(Xsub).T
        depth = random.randint(2,5)  
        tree = generate_tree(Xsub,Ysub, depth, None,features)
        if tree != None:
            list_trees.append(tree)
    return list_trees

def predict_tree(tree, xi):

    if tree.data is None:
        return tree.pred
    if tree.left==None and tree.right==None:
        return tree.pred    
    a = tree.data 
    feature,split = a[0],a[1]
    if xi[feature]<split:
        if tree.left !=None:
            val = predict_tree(tree.left,xi)
        else :
            val = predict_tree(tree.right,xi)
    if xi[feature]>=split:
        if tree.right !=None:
            val = predict_tree(tree.right,xi)
        else:
            val = predict_tree(tree.left,xi)
    return val

def predict(inputfile, modelfile):
    testX, testY, testIDs = loadfile(inputfile)
    list_trees = pickle.load(open(modelfile, "rb"))
    fp = open('tree_output.txt', 'w')
    predY=[]
    i = 0
    for xi in testX:
        predictions = [predict_tree(tree, xi) for tree in list_trees if tree !=None]
        pred = (Counter(predictions).most_common(1))[0][0]
        predY.append(pred)
        fp.write(str(testIDs[i]) + " " + str(pred) + "\n")
        i+=1
    accuracy = np.sum(predY==testY)/len(testY)
    fp.close()
    print("Accuracy  = " + str(accuracy * 100) + "%")


# X,Y,IDs = loadfile("train-data.txt")
# params = train(X.T, Y,500,0.05)
# pickle.dump( params, open( "model_txt.txt", "wb" ) )


#testX,testY,testIDs = loadfile("test-data.txt")
#list_trees = pickle.load( open( "model_txt.txt", "rb" ) )
#accuracy = predict(testX,testY, testIDs,list_trees)
#print("Accuracy  = " + str(accuracy*100) + "%")
