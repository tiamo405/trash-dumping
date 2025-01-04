# Instantiate ConvGRU model
import time
import numpy as np
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ConvGRUClassifier, FeatureExtractor


batch_size = 8
seq_length = 8
input_size = 2048
hidden_sizes = [512, 256]
n_layers = len(hidden_sizes)
output_size = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Model is running on", device)
model = ConvGRUClassifier(input_size=input_size, hidden_sizes=hidden_sizes, n_layers=n_layers, output_size=output_size).to(device)

# Load model
model.load_state_dict(torch.load('checkpoints/convGRU/model_convgru_weights_epoch_10.pth', map_location=torch.device('cpu')))
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