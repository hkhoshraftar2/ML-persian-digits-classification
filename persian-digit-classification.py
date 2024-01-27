import numpy as np
from scipy import io

import cv2
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np



#load dataset
dataset = io.loadmat('./dataset/Data_hoda_full.mat')

def load_hoda(training_sample_size=1000, test_sample_size=200, size=5):
    #load dataset
    trs = training_sample_size
    tes = test_sample_size

    #test and training set
    X_train_orginal = np.squeeze(dataset['Data'][:trs])
    y_train = np.squeeze(dataset['labels'][:trs])
    X_test_original = np.squeeze(dataset['Data'][trs:trs+tes])
    y_test = np.squeeze(dataset['labels'][trs:trs+tes])

    #resize
    X_train_5by5 = [cv2.resize(img, dsize=(size, size)) for img in X_train_orginal]
    X_test_5by_5 = [cv2.resize(img, dsize=(size, size)) for img in X_test_original]
    #reshape
    X_train = np.reshape(X_train_5by5, [-1,size**2])
    X_test = np.reshape(X_test_5by_5, [-1,size**2])
    
    return X_train, y_train, X_test, y_test

#test and training set
X_train_orginal = np.squeeze(dataset['Data'][:1000])
y_train = np.squeeze(dataset['labels'][:1000])
X_test_original = np.squeeze(dataset['Data'][1000:1200])
y_test = np.squeeze(dataset['labels'][1000:1200])

#resize
X_train_5by5 = [cv2.resize(img, dsize=(5, 5)) for img in X_train_orginal]
X_test_5by_5 = [cv2.resize(img, dsize=(5, 5)) for img in X_test_original]

#reshape
X_train = np.reshape(X_train_5by5, [-1,25])
X_test = np.reshape(X_test_5by_5, [-1,25])


X_train, y_train, X_test, y_test = load_hoda()
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                     weights='uniform')

#test model
print(neigh.predict([X_test[10]]))
sample = 24
X = [X_test[sample]]
predicted_class = neigh.predict(X)
print ("Sample {} is a {}, and you prediction is: {}.".format(sample, y_test[sample], predicted_class[0]))

#print probably neighbor values
print(neigh.predict_proba(X))

pred_classes = neigh.predict(X_test)
#np.mean(pred_classes == y_test)
acc = neigh.score(X_test, y_test)
print ("Accuracy is %.2f %%" %(acc*100))


