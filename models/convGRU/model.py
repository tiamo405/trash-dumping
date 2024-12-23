import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.out_gate = nn.Linear(input_size + hidden_size, hidden_size)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)


    def forward(self, input_, prev_state):

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [input_.size(0), self.hidden_size]
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).cuda()
            else:
                prev_state = torch.zeros(state_size)

        # concatenate input and previous state
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvGRU(nn.Module):

    def __init__(self, input_size, hidden_sizes, n_layers):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

        super(ConvGRU, self).__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            cell = ConvGRUCell(input_dim, self.hidden_sizes[i], kernel_size=3)
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells


    def forward(self, x, hidden=None):
        '''
        Parameters
        ----------
        x : 3D input tensor. (batch, seq_length, input_size).
        hidden : list of 2D hidden state representations. (batch, hidden_size).

        Returns
        -------
        upd_hidden : list of 2D hidden representation for each layer. (layer, batch, hidden_size).
        '''
        if not hidden:
            hidden = [None]*self.n_layers

        input_ = x

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = []
            for t in range(input_.size(1)):
                cell_hidden = cell(input_[:, t, :], cell_hidden)
                upd_cell_hidden.append(cell_hidden.unsqueeze(1))

            upd_cell_hidden = torch.cat(upd_cell_hidden, dim=1)
            upd_hidden.append(upd_cell_hidden[:, -1, :])

            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return upd_hidden

class ConvGRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_layers, output_size):
        super(ConvGRUClassifier, self).__init__()
        self.conv_gru = ConvGRU(input_size, hidden_sizes, n_layers)
        self.fc = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_size]
        batch_size, seq_len, input_size = x.shape

        hidden = None
        for t in range(seq_len):
            hidden = self.conv_gru(x[:, t, :].unsqueeze(1), hidden)

        # Use the final hidden state from the top layer
        final_hidden = hidden[-1]  # Shape: [batch_size, hidden_size]

        # Pass through fully connected layer
        output = self.fc(final_hidden)
        return output

import os
import cv2
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

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
        img_tensor = preprocess(img).unsqueeze(0).to(self.device)  # Add batch dimension
        with torch.no_grad():
            features = self.model(img_tensor)
        return features.flatten().cpu().numpy()


# 2. Dataset cho LSTM với chuỗi các ảnh
class SequenceDataset(Dataset):
    def __init__(self, img_size = 224, len_clip = 8):
        self.root_path = 'trash'
        self.feature_extractor = FeatureExtractor()
        self.sampling_rate = 1
        self.len_clip = len_clip
        self.img_size = img_size
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.file_names = []
        for label in ['Littering', 'Normal']:
            for folder in os.listdir(os.path.join('trash/labels', label)):
                self.file_names +=[os.path.join('trash/labels', label, folder, f) for f in os.listdir(os.path.join('trash/labels', label, folder))]
        #print(self.file_names)
        self.num_samples = len(self.file_names)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        txt_path = self.file_names[index].rstrip()
        img_split = txt_path.split('/')
        img_id = int(img_split[-1][:5])
        # image folder
        label_path = txt_path
        img_path = txt_path.replace('labels', 'rgb-images').replace('.txt', '.jpg')
        img_folder = os.path.join(self.root_path, 'rgb-images', img_split[2], img_split[3])
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
            path_tmp = os.path.join(self.root_path, 'rgb-images', img_split[2], img_split[3], '{:05d}.jpg'.format(img_id_temp))
            img = cv2.imread(path_tmp)

            bbox = np.loadtxt(label_path)
            left, top, right, bottom = int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4])
            img_person = img[top:bottom, left:right]
            feature = self.feature_extractor.extract_img(img_person)
            # img_person = self.preprocess(Image.fromarray(cv2.cvtColor(img_person, cv2.COLOR_BGR2RGB)))
            features.append(feature)

        label = 1 if 'Normal' in label_path else 0
        features_tensor = torch.tensor(features, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return features_tensor, label_tensor, bbox
# dataset = SequenceDataset()
# input_seq, target, _ = dataset[0]
# print(target)
if __name__ == '__main__':
    dataset = SequenceDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Instantiate ConvGRU model
    batch_size = 8
    seq_length = 8
    input_size = 2048
    hidden_sizes = [512, 256]
    n_layers = len(hidden_sizes)
    output_size = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Model is running on", device)
    model = ConvGRUClassifier(input_size=input_size, hidden_sizes=hidden_sizes, n_layers=n_layers, output_size=output_size).to(device)
    print(repr(model))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(10):
        model.train()
        for i,(input_seq, target, _) in enumerate(train_loader) :
            input_seq = input_seq.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output, 1)
            accuracy = (predicted == target).sum().item() / target.size(0)
            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}/10], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%")
        print(f"Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%")
        if os.path.isdir('checkpoints/convGRU') ==False:
            os.makedirs('checkpoints/convGRU')
        checkpoint_path = f"checkpoints/convGRU/model_weights_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        model.eval()
        test_loss = 0
        test_accuracy = 0
        with torch.no_grad():
            for input_seq, target,_ in test_loader:
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

