import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torch.optim as optim

class BasicCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self._initialize_weights()

        # Define the preprocessing steps
        self.preprocess = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to the input size the model expects
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize around the dataset mean and std
        ])

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Preprocess the input if it's a PIL Image
        if isinstance(x, Image.Image):
            if x.mode != 'RGB':
                x = x.convert('RGB')
            x = self.preprocess(x)
            x = x.unsqueeze(0)  # Add a batch dimension

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _set_weights(self, state_dict):
        self.load_state_dict(state_dict=state_dict)

    def get_gradients(self):
        gradients = {name: param.grad for name, param in self.named_parameters()}
        return gradients
    
    def set_gradients(self, averaged_gradients):
        for name, param in self.named_parameters():
            param.grad = averaged_gradients[name]
            
def initialize_model(model_weights):
    model = BasicCNN()
    model.load_state_dict(model_weights)
    optimizer = optim.SGD(model.parameters(), 0.001, 0.9)
    criterion = torch.nn.CrossEntropyLoss()
    return model, optimizer, criterion

def initialize_data(data_start, data_end):
     with open('./data/test') as file:
        data = file.read()
        return data
     
