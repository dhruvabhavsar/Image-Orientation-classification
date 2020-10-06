import numpy as np

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
                b=np.array(a[2:],dtype='int_')
                m=np.mean(b)
                sd=np.std(b)
                traindata.append((b - m)/sd)

    trainlabel=np.array(list(trainlabel),'int_')
    trainlabel=np.reshape(trainlabel,((-1, 1)))                   
    traindata=np.array(list(traindata),'int_')   
    return traindata,trainlabel,fname

# Training function
def train(trainfile, modelfile):
    # Create a copy of train file
    f = open(trainfile)
    with open(modelfile, 'w') as f1:
        for x in f.read().rstrip().split("\n"):
            f1.write(x+"\n")
    f.close()
    f1.close()

# Testing function   
def test(modelfile, testfile, outputfile):
    traindata, trainlabel, f = read_file(modelfile)
    testdata, testlabel, fname = read_file(testfile)
    
    k=25
    n = testdata.shape[0]
    pred = np.zeros((n,1))
    
    for i in range(n):
        occ = {}
        test = testdata[i,:].reshape(((-1, 1)))
        
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
            occ[labels[j]] = occurence[j]
        # Find the label with maximum occurences
        label = max(occ, key=occ.get)

        pred[i] = label
    
    # Find accuracy
    acc=0   
    for i in range(n):
        if pred[i]==testlabel[i]:
            acc+=1
    acc = (acc/pred.shape[0])*100
    
    print("Accuracy:",acc,"%")

    # Save output in file
    with open(outputfile, 'w') as f:
        for i in range(len(pred)):
            f.write("%s %d\n"%(fname[i],pred[i]))

# train("train-data.txt")
# test("nearest_file.txt","test-data.txt")
