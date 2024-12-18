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
    
    def extract_img(self, img):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = self.model(img_tensor)
        return features.flatten().numpy()

# 2. Dataset cho ConvGRU
class SequenceDataset(Dataset):
    def __init__(self, listtxt, img_size=224, len_clip=8):
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

        txt_path = self.file_names[index].rstrip()
        img_split = txt_path.split('/')
        img_id = int(img_split[-1][:5])
        label_path = os.path.join(self.root_path, img_split[0], img_split[1], img_split[2], '{:05d}.txt'.format(img_id))
        img_folder = os.path.join(self.root_path, 'rgb-images', img_split[1], img_split[2])
        max_num = len(os.listdir(img_folder))
        d = self.sampling_rate
        features = []
        
        for i in reversed(range(self.len_clip)):
            img_id_temp = img_id - i * d
            img_id_temp = max(1, min(img_id_temp, max_num))
            path_tmp = os.path.join(self.root_path, 'rgb-images', img_split[1], img_split[2], '{:05d}.jpg'.format(img_id_temp))
            img = cv2.imread(path_tmp)

            bbox = np.loadtxt(label_path)
            left, top, right, bottom = int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4])
            img_person = img[top:bottom, left:right]
            feature = self.feature_extractor.extract_img(img_person)
            features.append(feature)

        label = 0 if 'Normal' in label_path else 1

        return torch.tensor(features, dtype=torch.float32).unsqueeze(2), torch.tensor(label, dtype=torch.long)

# 3. ConvGRU Module
class ConvGRU(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRU, self).__init__()
        self.hidden_size = hidden_size
        padding = kernel_size // 2
        self.conv_xz = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        self.conv_hz = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding)
        self.conv_xr = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        self.conv_hr = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding)
        self.conv_xn = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        self.conv_hn = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding)

    def forward(self, x, hidden):
        r = torch.sigmoid(self.conv_xr(x) + self.conv_hr(hidden))
        z = torch.sigmoid(self.conv_xz(x) + self.conv_hz(hidden))
        n = torch.tanh(self.conv_xn(x) + r * self.conv_hn(hidden))
        hidden = (1 - z) * n + z * hidden
        return hidden

    def init_hidden(self, batch_size, spatial_dim):
        return torch.zeros(batch_size, self.hidden_size, spatial_dim[0], spatial_dim[1]).to(next(self.parameters()).device)

# 4. Mô hình ConvGRU
class ConvGRUModel(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_classes, kernel_size=3):
        super(ConvGRUModel, self).__init__()
        self.convgru = ConvGRU(input_channels, hidden_channels, kernel_size)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        hidden = self.convgru.init_hidden(batch_size, (height, width))

        for t in range(seq_len):
            hidden = self.convgru(x[:, t], hidden)

        pooled = nn.functional.adaptive_avg_pool2d(hidden, (1, 1)).view(batch_size, -1)
        output = self.fc(pooled)
        return output

# 5. Load toàn bộ dữ liệu trước khi train
listtext_train = 'trainlist.txt'
listtext_test = 'testlist.txt'

train_dataset = SequenceDataset(listtext_train)
test_dataset = SequenceDataset(listtext_test)

train_data = [train_dataset[i] for i in range(len(train_dataset))]
test_data = [test_dataset[i] for i in range(len(test_dataset))]

train_features = torch.stack([data[0] for data in train_data])
train_labels = torch.tensor([data[1] for data in train_data], dtype=torch.long)

test_features = torch.stack([data[0] for data in test_data])
test_labels = torch.tensor([data[1] for data in test_data], dtype=torch.long)

# 6. Huấn luyện mô hình
input_channels = 2048
hidden_channels = 128
num_classes = 2

model = ConvGRUModel(input_channels, hidden_channels, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for i in range(len(train_features)):
        features, labels = train_features[i].unsqueeze(0), train_labels[i].unsqueeze(0)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_features)
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    os.makedirs("checkpoints/convgru", exist_ok=True)
    checkpoint_path = f"checkpoints/convgru/model_weights_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model weights at {checkpoint_path}")

# 7. Đánh giá mô hình
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for i in range(len(test_features)):
        features, labels = test_features[i].unsqueeze(0), test_labels[i].unsqueeze(0)
        outputs = model(features)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Accuracy on test data:", accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds))
