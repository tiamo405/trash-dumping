import time
import torch
import numpy as np
import os


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import FeatureExtractor, LSTMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = LSTMModel(input_size=2048, hidden_size=128, output_size=2)
model.load_state_dict(torch.load('checkpoints/convLSTM/model_weights_epoch_9.pth', map_location=torch.device('cpu')))
model.eval()
model.to(device)

# fake 1 img for test
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
features = []
# check time
t0 = time.time()
feature = FeatureExtractor().extract_img(img)
for i in range(8):
    features.append(feature)
features = torch.tensor(features).unsqueeze(0).to(device)
with torch.no_grad():
    outputs = model(features)
print('Time: ', time.time() - t0)