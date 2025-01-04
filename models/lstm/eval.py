import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
import numpy as np
import os
import cv2

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import FeatureExtractor, SequenceDataset, LSTMModel

listtext_test = 'listtest.txt'
test_dataset = SequenceDataset(listtext_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = LSTMModel(input_size=2048, hidden_size=128, output_size=2)
model.load_state_dict(torch.load('checkpoints/convLSTM/model_weights_epoch_9.pth'))
model.eval()
model.to(device)

labels_true = []
labels_predic = []
with torch.no_grad():
    for features, labels in test_loader:
        labels_true.extend(labels.numpy())
        features = features.to(device)
        labels = labels.to(device)
        # check time 
        t0 = time.time()
        outputs = model(features)
        print('Time: ', time.time() - t0)
        _, predicted = torch.max(outputs, 1)
        labels_predic.extend(predicted.cpu().numpy())

print(labels_predic)