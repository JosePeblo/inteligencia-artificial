import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, random_split

import PIL
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

