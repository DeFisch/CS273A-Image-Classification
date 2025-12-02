#%%
from datasets import load_dataset
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

ds = load_dataset("zh-plus/tiny-imagenet")

# pick splits (adjust key names if needed)
train_split_name = "train"
val_split_name = "valid" if "valid" in ds else "validation"

hf_train = ds[train_split_name]
hf_val   = ds[val_split_name]

# ===========================
# 2. Transforms (ImageNet-style)
# ===========================
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # <-- add this
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

# ===========================
# 3. Wrap HF dataset as PyTorch Dataset
# ===========================
class HFTinyImageNetDataset(Dataset):
    def __init__(self, hf_ds, transform=None):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        img = example["image"]   # usually a PIL image
        label = example["label"] # integer class index

        if self.transform is not None:
            img = self.transform(img)

        return img, label

trainset = HFTinyImageNetDataset(hf_train, transform=transform)
testset  = HFTinyImageNetDataset(hf_val,   transform=transform)

print("Train size:", len(trainset))
print("Val size:", len(testset))
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
np.savez_compressed('imagenet_resnet18_features.npz',
                    train_features=train_features,
                    train_labels=train_labels,
                    test_features=test_features,
                    test_labels=test_labels)

print("Feature extraction completed and saved.")

# %%
# read back the features to verify
import numpy as np
data = np.load('imagenet_resnet18_features.npz')
print("Train features shape:", data['train_features'].shape)
print("Train labels shape:", data['train_labels'].shape)
print("Test features shape:", data['test_features'].shape)
print("Test labels shape:", data['test_labels'].shape)