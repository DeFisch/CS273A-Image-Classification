#%%
import numpy as np

# choose between 'cifar10_resnet18_features.npz' and 'imagenet_resnet18_features.npz'
DATA_PATH = 'cifar10_resnet18_features.npz'

data = np.load(DATA_PATH)
print("Train features shape:", data['train_features'].shape)
print("Train labels shape:", data['train_labels'].shape)
print("Test features shape:", data['test_features'].shape)
print("Test labels shape:", data['test_labels'].shape)

x_train = data['train_features']
y_train = data['train_labels']
x_test  = data['test_features']
y_test  = data['test_labels']

x_train = x_train[:5000]
y_train = y_train[:5000]

perm = np.random.permutation(len(x_train))
x_train = x_train[perm]
y_train = y_train[perm]

from sklearn import svm
clf = svm.SVC(kernel='rbf', C=10, gamma='scale')
clf.fit(x_train, y_train)
train_acc = clf.score(x_train, y_train)
test_acc  = clf.score(x_test, y_test)
print(f"SVM train accuracy: {train_acc*100:.2f}%, test accuracy: {test_acc*100:.2f}%")