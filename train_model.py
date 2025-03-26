import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
#from emnist import extract_training_samples, extract_test_samples
from scipy.io import loadmat
import numpy as np

#checking CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# x_train, y_train = extract_training_samples('byclass')
# x_test, y_test = extract_test_samples('byclass')

def load_emnist(path):
    data = loadmat(path)
    x_train = data['dataset']['train'][0][0]['images'][0][0].reshape(-1, 28, 28).transpose(0, 2, 1)  # Fix shape and transpose
    y_train = data['dataset']['train'][0][0]['labels'][0][0]
    x_test = data['dataset']['test'][0][0]['images'][0][0].reshape(-1, 28, 28).transpose(0, 2, 1)  # Fix shape and transpose
    y_test = data['dataset']['test'][0][0]['labels'][0][0]
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_emnist('emnist-byclass.mat')

#converting to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1) / 255.0
y_train = torch.tensor(y_train, dtype=torch.long).squeeze()
x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1) / 255.0
y_test = torch.tensor(y_test, dtype=torch.long).squeeze()

#creating datasets
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

#convolutional neural network model
class CNNTrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 62)
        )
    
    def forward(self, x):
        return self.net(x)

model = CNNTrain().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#using dataloaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

#training loop
for epoch in range(10):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    #evaluation on test data
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Epoch {epoch+1}, Accuracy: {100*correct/total:.2f}%")

#save model
torch.save(model.state_dict(), 'emnist_model.pth')