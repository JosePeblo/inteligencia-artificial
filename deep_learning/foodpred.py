import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import threading

NUM_CLASES = 15
CLASSES = [
    'apple_pie', 'baked_potato', 'burger', 'cheesecake', 'crispy_chicken', 
    'donut', 'fried_rice', 'fries', 'hot_dog', 'ice_cream', 'omelette', 
    'pizza', 'sandwich', 'sushi', 'taco'
]

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model = torchvision.models.resnet50()

model.fc = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Linear(256, NUM_CLASES),
)

model.load_state_dict(torch.load(Path('FoodClassModel2.pt'), weights_only=True))

model.eval()

def MakePrediction(image: Image):
    imgTensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        pred = model(imgTensor)
        prob = torch.softmax(pred, dim=1)

        predClass = torch.argmax(prob, dim=1).item()
        confidence = prob[0, predClass].item() * 100

    return (predClass, confidence)

def MakePredictionTask(frame, result, event):    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(img)

    result[0] = MakePrediction(img)

    event.set()


def StartCam():
    cam = cv2.VideoCapture(0)

    result = [None]
    event = threading.Event()
    event.set()

    predThread = None

    while True:
        _, img = cam.read()

        if event.is_set():
            # predClass = result[0]

            event.clear()

            predThread = threading.Thread(
                target=MakePredictionTask, 
                args=(img, result, event)
            )

            predThread.start()
        
        img = cv2.flip(img, 1)
        if result[0] is not None:
            (predClass, confidence) = result[0]
            cv2.putText(img, f'{CLASSES[predClass]} {confidence:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('I am hungry', img)

        key = cv2.waitKey(1) & 0xFF

        if key == 27: # Esc key
            break

        if cv2.getWindowProperty('I am hungry', cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()

parser = argparse.ArgumentParser(
    prog='foodpred.py',
    description='This program preddicts food wit AI!!',
)

parser.add_argument('-i', '--image', help='path of the image to predict')
parser.add_argument('-p', '--plot', action='store_true', help='plot the result')
parser.add_argument('-v', '--video', action='store_true', help='start video mode')

args = parser.parse_args()

if(args.image):
    image = Image.open(Path(args.image))
    
    (predClass, confidence) = MakePrediction(image)

    if(args.plot):
        plt.title(f'{CLASSES[predClass]} {confidence:.2f}%')
        plt.imshow(image)
        plt.show()
    else:
        print(f'{CLASSES[predClass]} {confidence:.2f}%')

elif(args.video):
    StartCam()




