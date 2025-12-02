#%%
import torchvision.datasets as datasets
import torchvision.transforms as transforms

trainset = datasets.CIFAR10(root='./data', train=True, download=True,
                            transform=transforms.ToTensor())
testset  = datasets.CIFAR10(root='./data', train=False, download=True,
                            transform=transforms.ToTensor())

# %%
from matplotlib import pyplot as plt
import numpy as np

img = trainset[0][0]  # get the first image tensor from the training set
img = np.transpose(img.numpy(), (1, 2, 0))  # convert to HWC format for plotting
plt.imshow(img)
plt.show()
#%%
import timm
model = timm.create_model("resnet18", pretrained=True, num_classes=0)
model.eval()
model = model.cuda()
# %%
# preprocess all the features and save them
import torch
from tqdm import tqdm

train_loader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=False)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)

def extract_features(loader, model):
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader):
            imgs = imgs.to('cuda')
            feats = model(imgs)                 
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


train_features, train_labels = extract_features(train_loader, model)
test_features, test_labels = extract_features(test_loader, model)
np.savez_compressed('cifar10_resnet18_features.npz',
                    train_features=train_features,
                    train_labels=train_labels,
                    test_features=test_features,
                    test_labels=test_labels)

print("Feature extraction completed and saved.")