import pickle as pkl
import numpy as np
from sklearn import metrics
from sklearn import svm

features = pkl.load(open('training_mfcc_features_noise_final','rb'))
labels = pkl.load(open('training_mfcc_labels_noise_final','rb'))
features_val = pkl.load(open('validation_mfcc_features_noise_final','rb'))
labels_val = pkl.load(open('validation_mfcc_labels_noise_final','rb'))
clf = svm.SVC(kernel = 'linear',verbose = True)
features = np.asarray(features).reshape(10000,-1)
print(features.shape)
clf.fit(features,labels)
features_val = np.asarray(features_val).reshape(2494,-1)
pred = clf.predict(features_val)

print(pred)

print(metrics.classification_report(labels_val,pred))
print(metrics.accuracy_score(labels_val,pred))
print(metrics.confusion_matrix(labels_val,pred))

pkl.dump(clf,open('clf_mfcc_noise_final','wb'))