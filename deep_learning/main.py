import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam

from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        img_tensor, label = torch.load(self.files[idx])
        return img_tensor, label

dataset = FoodDataset(PREPROC_PATH)

# Perform the split into training and validation
trainSize = int(0.8 * len(dataset))
valSize = len(dataset) - trainSize
trainDataset, valDataset = random_split(dataset, [trainSize, valSize])


# Create the loaders
trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)
valLoader = DataLoader(valDataset, batch_size=64, shuffle=False)

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

# Define the model
IMG_CHANNELS = 3
kernel_size = 3

FLATTENED_SIZE = 256 * 14 * 14

model = nn.Sequential(
    nn.Conv2d(IMG_CHANNELS, 32, kernel_size, stride=1, padding=1), # 32 x 224 x 224
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2), # 32 x 112 x 112

    nn.Conv2d(32, 64, kernel_size, stride=1, padding=1), # 64 x 112 x 112
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.MaxPool2d(2, stride=2), # 64 x 56 x 56

    nn.Conv2d(64, 128, kernel_size, stride=1, padding=1), # 128 x 56 x 56
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2), # 128 x 28 x 28

    nn.Conv2d(128, 256, kernel_size, stride=1, padding=1), # 256 x 28 x 28
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2), # 256 x 14 x 14


    nn.Flatten(),

    nn.Linear(FLATTENED_SIZE, 256),
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(256, 256),
    nn.ReLU(),

    nn.Linear(256, NUM_CLASES)
)

model.to(device)
print(model)

lossFunc = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

def getBatchAccuracy(output, y):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct


def validate():
    val_loss = 0
    correct = 0
    model.eval()

    with torch.no_grad():
        for x, y in valLoader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            val_loss += lossFunc(output, y).item()
            correct += getBatchAccuracy(output, y)

    avg_loss = val_loss / len(valLoader)
    accuracy = correct / valSize

    print(f'(val) Loss: {avg_loss:.4f} Accuracy: {accuracy:.4f}')

def train():
    train_loss = 0
    correct = 0
    model.train()
    
    for x, y in trainLoader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        optimizer.zero_grad()
        batch_loss = lossFunc(output, y)
        batch_loss.backward()
        optimizer.step()

        train_loss += batch_loss.item()


        correct += getBatchAccuracy(output, y)

    avg_loss = train_loss / len(trainLoader)
    accuracy = correct / trainSize

    print(f'(train) Loss: {avg_loss:.4f} Accuracy: {accuracy:.4f}')

epochs = 100

# Train the model
for epoch in range(epochs):
    print(f'Epoch {epoch}')
    train()
    validate()


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

# Save the model
torch.save(model.state_dict(), Path('./FCModel.pth'))