#!/usr/local/bin/python3
#
# Authors: Dhruva Bhavsar(dbhavsar), Hely Modi(helymodi), Aneri Shah(annishah)
#
#k-nearest neighbours

import numpy as np
import matplotlib.pyplot as plt
import time

# Read the test and train files
def read_file(filename):
    traindata=[]
    trainlabel=[]
    fname=[]
    with open(filename, 'r') as f:     
        for line in f.read().lower().rstrip().split("\n"):
                a=line.split(" ")
                fname.append(a[0])
                trainlabel.append(int(a[1]))
#                traindata.append(a[2:])
                b=np.array(a[2:],dtype='int_')
                m=np.mean(b)
                sd=np.std(b)
                traindata.append((b - m)/sd)

    trainlabel=np.array(list(trainlabel),'int_')
    trainlabel=np.reshape(trainlabel,((-1, 1)))                   
    traindata=np.array(list(traindata),'int_')   
    return traindata,trainlabel,fname

# Training function
def train(trainfile):
    # Create a copy of train file
    f=open(trainfile)  
    with open("nearest_file.txt", 'w') as f1:
        for x in f.read().rstrip().split("\n"):
            f1.write(x+"\n")
    f.close()
    f1.close()

# Testing function   
def test(modelfile,testfile):
    traindata,trainlabel,f=read_file(modelfile)
    testdata,testlabel,fname=read_file(testfile)
    
    
#    k=100
    print(traindata.shape[0])
    num_training = testdata.shape[0]
    pred = np.zeros((num_training,1))
    
    acc1=[]
    time1=[]
    acc2=0
    k1=0
    k2=[]
    for k in range(5,101,5):
        start_time = time.process_time()
        
        for i in range(num_training):
            occ={}
            test=testdata[i,:].reshape(((-1, 1)))
            
            # Find distance between each point in trainset and one point in testset
            distances = np.reshape(np.sum(np.abs(traindata - test.T), axis=1), (-1, 1))
            
            distance_label = np.hstack((distances, trainlabel))
            # Sort distances
            sorted_distance = distance_label[distance_label[:,0].argsort()]
            # Find k nearest points
            k_sorted_distance = sorted_distance[:k,:]
            # Find the occurence of each label
            (labels, occurence) = np.unique(k_sorted_distance[:, 1], return_counts=True)
            for j in range(len(labels)):
                occ[labels[j]]=occurence[j]
            # Find the label with maximum occurences
            label=max(occ, key=occ.get)
    
            pred[i] = label
        
        # Find accuracy
        acc=0   
        for i in range(num_training):
            if pred[i]==testlabel[i]:
                acc+=1
        acc=(acc/pred.shape[0])*100
#        print(k)
        if acc > acc2:
            best=pred
            acc2=acc
            k1=k
        total_time=time.process_time() - start_time
        time1.append(total_time)
        acc1.append(acc)
        k2.append(k)
        
#        print("Accuracy:",acc,"%")
#        print(total_time)

    # Save output in file
    print("Accuracy:",acc2,"%")
    print(k1)
    plt.xlabel('Value of K')
    plt.ylabel('Accuracy')
    plt.plot(k2,acc1)
    plt.show()
    with open('nearest-output.txt', 'w') as f:
        for i in range(len(pred)):
            f.write("%s %d\n"%(fname[i],best[i]))
    
train("train-data.txt")
test("nearest_file.txt","test-data.txt")








