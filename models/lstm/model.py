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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove last layer
        self.model = self.model.to(self.device)
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
        img_tensor = img_tensor.to(self.device) # Chuyển sang GPU nếu có
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

        txt_path = self.file_names[index].rstrip()
        img_split = txt_path.split('/')
        img_id = int(img_split[-1][:5])
        # image folder
        label_path = os.path.join(self.root_path, img_split[0], img_split[1], img_split[2], '{:05d}.txt'.format(img_id))
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
            img = cv2.imread(path_tmp)
            
            bbox = np.loadtxt(label_path)
            left, top, right, bottom = int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4])
            img_person = img[top:bottom, left:right]
            feature = self.feature_extractor.extract_img(img_person)
            features.append(feature)

        label = 1 if 'Normal' in label_path else 0

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

if 'name' == '__main__':
    input_size = 2048  # Dựa trên ResNet50 output
    hidden_size = 128
    output_size = 2

    model = LSTMModel(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Số lượng epoch
    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Tính toán độ chính xác (accuracy)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Cộng dồn loss cho epoch
            total_loss += loss.item()
            
            # Cập nhật gradient và tối ưu hóa
            loss.backward()
            optimizer.step()
        
        # Tính toán loss và accuracy trung bình cho epoch
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        # Lưu trọng số của mô hình sau mỗi epoch (hoặc ở epoch cuối)
        if os.path.isdir("checkpoints/lstm") == False:
            os.makedirs("checkpoints/lstm")
        checkpoint_path = f"checkpoints/lstm/model_weights_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved model weights at {checkpoint_path}")

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
