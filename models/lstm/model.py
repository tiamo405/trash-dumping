import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from PIL import Image
from typing import List

# 1. Hàm trích xuất đặc trưng từ ảnh
class FeatureExtractor:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove last layer
        self.model.eval()
    
    def extract(self, img_path: str):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = self.model(img_tensor)
        return features.flatten().numpy()
    def extract_img(self, img):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # img cv2 convert to PIL
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = self.model(img_tensor)
        return features.flatten().numpy()

# 2. Dataset cho LSTM với chuỗi các ảnh
class SequenceDataset(Dataset):
    def __init__(self, listtxt, img_size = 224, len_clip = 8):
        self.listtxt = listtxt
        self.root_path = 'trash'
        self.feature_extractor = FeatureExtractor()
        self.sampling_rate = 1
        self.len_clip = len_clip
        self.img_size = img_size

        with open(os.path.join(self.root_path, self.listtxt), 'r') as f:
            self.file_names = f.readlines()
        self.num_samples = len(self.file_names)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        label_path = self.file_names[index].rstrip()
        img_split = label_path.split('/')
        img_id = int(img_split[-1][:5])
        # image folder
        img_folder = os.path.join(self.root_path, 'rgb-images', img_split[1], img_split[2])
        max_num = len(os.listdir(img_folder))
        d = self.sampling_rate
        video_clip = []
        features = []
        for i in reversed(range(self.len_clip)):
            img_id_temp = img_id - i * d # lay cac hinh anh cach nhau d frame
            if img_id_temp < 1:
                img_id_temp = 1
            elif img_id_temp > max_num:
                img_id_temp = max_num
            path_tmp = os.path.join(self.root_path, 'rgb-images', img_split[1], img_split[2], '{:05d}.jpg'.format(img_id_temp))
            path_tmp_fail = os.path.join(self.root_path, 'rgb-images', img_split[1], img_split[2] ,'{:05d}.jpg'.format(img_id))
            try:
                img = cv2.imread(path_tmp)
                feature = self.feature_extractor.extract_img(img)
            except:
                img = cv2.imread(path_tmp_fail)
                feature = self.feature_extractor.extract_img(img)
            features.append(feature)


        label = 0 if 'Normal' in label_path else 1

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# read dataset
listtext_train = 'trainlist.txt'
listtext_test = 'testlist.txt'
train_dataset = SequenceDataset(listtext_train)
test_dataset = SequenceDataset(listtext_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 3. Mô hình LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Lấy output của bước cuối cùng
        return x

input_size = 2048  # Dựa trên ResNet50 output
hidden_size = 128
output_size = 2

model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train loop
epochs = 10
for epoch in range(epochs):
    model.train()
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Đánh giá mô hình
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# Báo cáo kết quả
print("Accuracy:", accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds))
