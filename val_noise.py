import pickle as pkl
import numpy as np
from sklearn import metrics
from sklearn import svm

features = pkl.load(open('training_spectograms_features_noise_final','rb'))
labels = pkl.load(open('training_spectograms_labels_noise_final','rb'))
features_val = pkl.load(open('validation_spectograms_features_noise_final','rb'))
labels_val = pkl.load(open('validation_spectograms_labels_noise_final','rb'))
clf = svm.SVC(kernel = 'linear', verbose = True)

print('hi')

features = np.asarray(features).reshape(10000,-1)
features_val = np.asarray(features_val).reshape(2494,-1)
print(features.shape,features_val.shape)

clf.fit(features,labels)
pkl.dump(clf,open('clf_spec_noise_final','wb'))
pred = clf.predict(features_val)


print(pred)
print(metrics.classification_report(labels_val,pred))
print(metrics.accuracy_score(labels_val,pred))
print(metrics.confusion_matrix(labels_val,pred))
pkl.dump(clf,open('clf_spec_noise_final','wb'))