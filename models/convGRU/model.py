###################################################
# Nicolo Savioli, 2021 -- Conv-GRU pytorch v 1.1  #
###################################################

import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
import os
import cv2
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
        img_tensor = preprocess(img).unsqueeze(0).to(self.device)  # Add batch dimension
        with torch.no_grad():
            features = self.model(img_tensor)
        return features.flatten().cpu().numpy()
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
        return features.flatten().cpu().numpy()

# 2. Dataset cho LSTM với chuỗi các ảnh
class SequenceDataset(Dataset):
    def __init__(self, listtxt, img_size = 224, len_clip = 8):
        self.listtxt = listtxt
        self.root_path = 'trash'
        # self.feature_extractor = FeatureExtractor()
        self.sampling_rate = 1
        self.len_clip = len_clip
        self.img_size = img_size
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
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
            # feature = self.feature_extractor.extract_img(img_person)
            img_person = self.preprocess(Image.fromarray(cv2.cvtColor(img_person, cv2.COLOR_BGR2RGB)))
            features.append(img_person.unsqueeze(0))

        label = 1 if 'Normal' in label_path else 0
        features_tensor = torch.cat(features, dim=0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return features_tensor, label_tensor

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, cuda_flag):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.ConvGates = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size, 3, padding=self.kernel_size // 2)
        self.Conv_ct = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, 3, padding=self.kernel_size // 2)

    def forward(self, input, hidden):
        if hidden is None:
            size_h = [input.data.size()[0], self.hidden_size] + list(input.data.size()[2:])
            if self.cuda_flag:
                hidden = Variable(torch.zeros(size_h)).cuda()
            else:
                hidden = Variable(torch.zeros(size_h))
        c1 = self.ConvGates(torch.cat((input, hidden), 1))
        (rt, ut) = c1.chunk(2, 1)
        reset_gate = f.sigmoid(rt)
        update_gate = f.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate, hidden)
        p1 = self.Conv_ct(torch.cat((input, gated_hidden), 1))
        ct = f.tanh(p1)
        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        return next_h

class ConvGRUClassifier(nn.Module):
    def __init__(self, input_channels, hidden_size, kernel_size, num_classes, seq_length, cuda_flag):
        super(ConvGRUClassifier, self).__init__()
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.conv_gru = ConvGRUCell(input_channels, hidden_size, kernel_size, cuda_flag)
        self.fc1 = nn.Linear(hidden_size * 224 * 224, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
    def forward(self, x):
        batch_size = x.size(0)
        h_next = None
        for t in range(self.seq_length):
            h_next = self.conv_gru(x[:, t], h_next)
        h_next_flat = h_next.view(batch_size, -1)
        output = self.fc1(h_next_flat)
        output = self.fc2(output)
        return output

def test(model, criterion, input_seq, target, device):
    model.eval()
    input_seq = input_seq.to(device)
    target = target.to(device)
    with torch.no_grad():
        output = model(input_seq)
        loss = criterion(output, target)
        _, predicted = torch.max(output, 1)
        accuracy = (predicted == target).sum().item() / target.size(0)
        print(f"Test Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%")

def main():
    listtext_train = 'trainlist.txt'
    listtext_test = 'testlist.txt'
    train_dataset = SequenceDataset(listtext_train)
    test_dataset = SequenceDataset(listtext_test)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    seq_length = 8
    input_channels = 3
    hidden_size = 64
    kernel_size = 3
    num_classes = 2
    cuda_flag = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_flag else 'cpu')

    print('Initializing ConvGRUClassifier:')
    model = ConvGRUClassifier(input_channels, hidden_size, kernel_size, num_classes, seq_length, cuda_flag)
    model.to(device)
    print(repr(model))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        model.train()
        for input_seq, target in train_loader:
            input_seq = input_seq.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output, 1)
            accuracy = (predicted == target).sum().item() / target.size(0)
        print(f"Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%")
        if os.path.isdir("checkpoints/convgru") == False:
            os.makedirs("checkpoints/convgru")
        checkpoint_path = f"checkpoints/convgru/model_weights_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        # Test
        model.eval()
        test_loss = 0
        test_accuracy = 0
        with torch.no_grad():
            for input_seq, target in test_loader:
                input_seq = input_seq.to(device)
                target = target.to(device)
                output = model(input_seq)
                loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = torch.max(output, 1)
                test_accuracy += (predicted == target).sum().item()
        test_loss /= len(test_loader)
        test_accuracy /= len(test_dataset)
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
