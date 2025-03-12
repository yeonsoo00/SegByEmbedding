import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

def load_data(path2img):
    """
    Load image
    """
    image = cv2.imread(path2img, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
    return image

def generate_patches_w_label(image, )
