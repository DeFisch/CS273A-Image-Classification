#%%
import numpy as np
from joblib import Parallel, delayed
from sklearn import svm

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

# define different training set sizes to experiment with
data_used = np.linspace(1000, len(x_train), 10, dtype=int)

perm = np.random.permutation(len(x_train))
x_train = x_train[perm]
y_train = y_train[perm]

def train_and_eval(n_samples):
    """Train one SVM on the first n_samples and return metrics."""
    clf = svm.SVC(kernel='rbf', C=10, gamma='scale')
    clf.fit(x_train[:n_samples], y_train[:n_samples])
    train_acc = clf.score(x_train[:n_samples], y_train[:n_samples])
    test_acc  = clf.score(x_test, y_test)
    return n_samples, train_acc, test_acc

# parallelize over different training set sizes
results = Parallel(n_jobs=-1, prefer="processes")(
    delayed(train_and_eval)(n) for n in data_used
)

# print results
for n_samples, train_acc, test_acc in results:
    print(f"{n_samples:6d} samples -> "
          f"train acc: {train_acc*100:5.2f}%, "
          f"test acc: {test_acc*100:5.2f}%")

#%%
# plot results
import matplotlib.pyplot as plt

n_samples_list = [r[0] for r in results]
train_acc_list = [r[1] for r in results]
test_acc_list = [r[2] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(n_samples_list, train_acc_list, label='Train Accuracy', marker='o')
plt.plot(n_samples_list, test_acc_list, label='Test Accuracy', marker='s')
plt.xlabel('Number of Training Samples')
plt.ylabel('Accuracy')
plt.title('SVM Accuracy vs Number of Training Samples')
plt.legend()
# save as pdf
plt.savefig('svm_accuracy_vs_samples.pdf')
plt.show()

#%%
from sklearn import svm
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt

# Train with different SVM kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Define soft vs hard margin via different C values
# (soft: smaller C, hard: large C approximating hard-margin)
margin_configs = {
    "soft": 1.0,
    "hard": 1e3,
}

# choose 5000 samples for kernel comparison
perm = np.random.permutation(len(x_train))
x_train = x_train[perm]
y_train = y_train[perm]

x_train_subset = x_train[:5000]
y_train_subset = y_train[:5000]

def train_and_eval_kernel(kernel_name, margin_name, C_val):
    """Train one SVM with a specific kernel and margin (C) and return metrics."""
    clf = svm.SVC(kernel=kernel_name, C=C_val, gamma='scale')
    clf.fit(x_train_subset, y_train_subset)
    train_acc = clf.score(x_train_subset, y_train_subset)
    test_acc  = clf.score(x_test, y_test)
    return kernel_name, margin_name, C_val, train_acc, test_acc

# parallelize over different SVM kernels and margin types
results = Parallel(n_jobs=-1, prefer="processes")(
    delayed(train_and_eval_kernel)(k, m_name, C_val)
    for k in kernels
    for m_name, C_val in margin_configs.items()
)

# print results
for kernel_name, margin_name, C_val, train_acc, test_acc in results:
    print(f"{kernel_name:8s} ({margin_name:4s}, C={C_val:7.1f}) -> "
          f"train acc: {train_acc*100:5.2f}%, "
          f"test acc: {test_acc*100:5.2f}%")

# prepare data for plotting
labels = [f"{k}-{m}" for (k, m, _, _, _) in results]
train_accs = [r[3] for r in results]
test_accs  = [r[4] for r in results]

x = np.arange(len(labels))
width = 0.35

#%%
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, train_accs, width, label='Train Accuracy')
plt.bar(x + width/2, test_accs,  width, label='Test Accuracy')
plt.xticks(x, labels, rotation=45)
plt.xlabel('SVM Kernel / Margin Type')
plt.ylabel('Accuracy')
plt.title('SVM Accuracy for Different Kernels and Margin Types (5000 samples)')
plt.legend()
plt.tight_layout()
plt.savefig('svm_kernel_margin_comparison.pdf')
plt.show()
