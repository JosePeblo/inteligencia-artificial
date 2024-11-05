import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import torchvision

from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

DATASET_PATH = Path('D:/Dev/Datasets/FoodClassification')
PREPROC_PATH = Path('D:/Dev/Datasets/PreprocFoodClassification')
PREPROC_PATH.mkdir(parents=True, exist_ok=True)

# Transformations for the images
transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

originalDataset = datasets.ImageFolder(root=DATASET_PATH)

# Preprocess the dataset and save it to disk to save time loading from disk
for idx, (img, label) in enumerate(tqdm(originalDataset, desc="Preprocessing images")):
    imgTensor = transforms(img)
    
    # Get the class folder path in preprocessed directory
    className = originalDataset.classes[label]
    classFolder = PREPROC_PATH / className
    classFolder.mkdir(parents=True, exist_ok=True)
    
    # Save tensor as a .pt file
    tensor_path = classFolder / f"{idx}.pt"
    torch.save((imgTensor, label), tensor_path)

CLASSES = originalDataset.classes
NUM_CLASES = len(CLASSES)

print(NUM_CLASES, CLASSES)

# Custom datasert to load the preprocessed data
class FoodDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.files = list(self.root.glob("**/*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_tensor, label = torch.load(self.files[idx], weights_only=True)
        return img_tensor, label

dataset = FoodDataset(PREPROC_PATH)

# Perform the split into training and validation
trainSize = int(0.8 * len(dataset))
valSize = len(dataset) - trainSize
trainDataset, valDataset = random_split(dataset, [trainSize, valSize])

# Create the loaders
trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True, pin_memory=True)
valLoader = DataLoader(valDataset, batch_size=32, shuffle=False, pin_memory=True)

# Plot some of the images and their classes
images, labels = next(iter(trainLoader))
images = images.cpu().numpy()
labels = labels.cpu().numpy()

fig, axes = plt.subplots(4, 4, figsize=(10, 10)) 
axes = axes.flatten()

for ax, img, label in zip(axes, images, labels):
    img = (img.transpose(1, 2, 0) + 1) / 2
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(CLASSES[label])

plt.tight_layout()
plt.show()

class Bottleneck(nn.Module):
    def __init__(self, inChan, outChan, expansion=4, stride=1):
        super().__init__()

        midChan = outChan // expansion

        self.convBlock = nn.Sequential(
            nn.Conv2d(inChan, midChan, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(midChan),
            nn.ReLU(inplace=True),

            nn.Conv2d(midChan, midChan, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(midChan),
            nn.ReLU(inplace=True),

            nn.Conv2d(midChan, outChan, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outChan),
        )

        self.downsample = None

        if inChan != outChan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(inChan, outChan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outChan)
            ) 

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.convBlock(x)

        skip = x
        if self.downsample:
            skip = self.downsample(x)

        out += skip
        out = self.relu(out)

        return out


# Kinda Resnet50
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, stride=2),

    # Stage 1
    Bottleneck(64, 256, stride=1),
    Bottleneck(256, 256, stride=1),
    Bottleneck(256, 256, stride=1),

    # Stage 2
    Bottleneck(256, 512, stride=2),
    Bottleneck(512, 512, stride=1),
    Bottleneck(512, 512, stride=1),
    Bottleneck(512, 512, stride=1),

    # Stage 3
    Bottleneck(512, 1024, stride=2),
    Bottleneck(1024, 1024, stride=1),
    Bottleneck(1024, 1024, stride=1),
    Bottleneck(1024, 1024, stride=1),
    Bottleneck(1024, 1024, stride=1),
    Bottleneck(1024, 1024, stride=1),

    # Stage 4
    Bottleneck(1024, 2048, stride=2),
    Bottleneck(2048, 2048, stride=1),
    Bottleneck(2048, 2048, stride=1),

    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
    nn.Flatten(),
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Linear(256, NUM_CLASES),
    nn.Sigmoid(),
)

model.to(device)
print(model)

lossFunc = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

trainLosses = []
trainAcc = []
valLosses = []
valAcc = []

bestLoss = float('inf')
bestModelWeights = None
patience = 10

def getBatchAccuracy(output, y):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / len(y)

def validate():
    valLoss = 0
    correct = 0
    model.eval()

    with torch.no_grad():
        for x, y in valLoader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            valLoss += lossFunc(output, y).item() * x.size(0)
            correct += getBatchAccuracy(output, y) * x.size(0)

    avgLoss = valLoss / valSize
    accuracy = correct / valSize

    valLosses.append(avgLoss)
    valAcc.append(accuracy)

    print(f'(val) Loss: {avgLoss:.4f} Accuracy: {accuracy:.4f}')

    global bestLoss, bestModelWeights, patience

    if valLoss < bestLoss:
        bestLoss = valLoss
        bestModelWeights = copy.deepcopy(model.state_dict())
        patience = 10

    else:
        patience -= 1
        if patience == 0:
            return True
        
    return False

def train():
    trainLoss = 0
    correct = 0
    model.train()
    
    for x, y in trainLoader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        optimizer.zero_grad()
        batchLoss = lossFunc(output, y)
        batchLoss.backward()
        optimizer.step()

        trainLoss += batchLoss.item() * x.size(0)
        correct += getBatchAccuracy(output, y) * x.size(0)

    avgLoss = trainLoss / trainSize
    accuracy = correct / trainSize

    trainLosses.append(avgLoss)
    trainAcc.append(accuracy)

    print(f'(train) Loss: {avgLoss:.4f} Accuracy: {accuracy:.4f}')


epochs = 100

# Train the model
for epoch in range(epochs):
    print(f'Epoch {epoch}')
    train()
    if validate(): 
        break # Early stopping
    
    scheduler.step()


model.load_state_dict(bestModelWeights)


# Print the model predictions
model.eval()
images, labels = next(iter(valLoader))

images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

images = images.cpu().numpy()
labels = labels.cpu().numpy()
preds = preds.cpu().numpy()

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.flatten()

for ax, img, label, pred in zip(axes, images, labels, preds):
    img = (img.transpose(1, 2, 0) + 1) / 2
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'True: {CLASSES[label]}\nPred: {CLASSES[pred]}')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(trainLosses, label='Training Loss')
plt.plot(valLosses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(trainAcc, label='Training Accuracy')
plt.plot(valAcc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
torch.save(model.state_dict(), Path('../prototypes/FCModel.pth'))

# Run with transfer learning

# Load the model
model = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.DEFAULT)

model.fc = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Linear(256, NUM_CLASES),
    nn.Sigmoid(),
)

for param in model.fc.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True


model.to(device)
print(model)

lossFunc = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

trainLosses = []
trainAcc = []
valLosses = []
valAcc = []

bestLoss = float('inf')
bestModelWeights = None
patience = 10

def getBatchAccuracy(output, y):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / len(y)

def validate():
    valLoss = 0
    correct = 0
    model.eval()

    with torch.no_grad():
        for x, y in valLoader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            valLoss += lossFunc(output, y).item() * x.size(0)
            correct += getBatchAccuracy(output, y) * x.size(0)

    avgLoss = valLoss / valSize
    accuracy = correct / valSize

    valLosses.append(avgLoss)
    valAcc.append(accuracy)

    print(f'(val) Loss: {avgLoss:.4f} Accuracy: {accuracy:.4f}')

    global bestLoss, bestModelWeights, patience

    if valLoss < bestLoss:
        bestLoss = valLoss
        bestModelWeights = copy.deepcopy(model.state_dict())
        patience = 10

    else:
        patience -= 1
        if patience == 0:
            return True
        
    return False

def train():
    trainLoss = 0
    correct = 0
    model.train()
    
    for x, y in trainLoader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        optimizer.zero_grad()
        batchLoss = lossFunc(output, y)
        batchLoss.backward()
        optimizer.step()

        trainLoss += batchLoss.item() * x.size(0)
        correct += getBatchAccuracy(output, y) * x.size(0)

    avgLoss = trainLoss / trainSize
    accuracy = correct / trainSize

    trainLosses.append(avgLoss)
    trainAcc.append(accuracy)

    print(f'(train) Loss: {avgLoss:.4f} Accuracy: {accuracy:.4f}')


epochs = 30

# Train the model
for epoch in range(epochs):
    print(f'Epoch {epoch}')
    train()
    if validate(): 
        break # Early stopping
    
    scheduler.step()


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(trainLosses, label='Training Loss')
plt.plot(valLosses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(trainAcc, label='Training Accuracy')
plt.plot(valAcc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# Print the model predictions
model.eval()
images, labels = next(iter(valLoader))

images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

images = images.cpu().numpy()
labels = labels.cpu().numpy()
preds = preds.cpu().numpy()

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.flatten()

for ax, img, label, pred in zip(axes, images, labels, preds):
    img = (img.transpose(1, 2, 0) + 1) / 2
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'True: {CLASSES[label]}\nPred: {CLASSES[pred]}')

plt.tight_layout()
plt.show()