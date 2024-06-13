import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, video_list, labels, transform=None):
        self.video_list = video_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video = self.video_list[idx]
        label = self.labels[idx]
        if self.transform:
            video = self.transform(video)
        return video, label

class I3D(nn.Module):
    def __init__(self, num_classes=10):
        super(I3D, self).__init__()
        self.conv3d = nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Temporary dummy forward pass to find the output shape
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 112, 112)
            dummy_output = self.pool(F.relu(self.conv3d(dummy_input)))
            self.fc_input_size = int(np.prod(dummy_output.size()))
        
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        x = x.float()  # Convert input tensor to float
        x = self.pool(F.relu(self.conv3d(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create dummy data
video_list = [np.random.rand(3, 32, 112, 112).astype(np.float32) for _ in range(8)]  # 8 videos, 32 frames, 112x112 resolution
labels = np.random.randint(0, 10, 8).astype(np.int64)  # Ensure labels are of type int64

dataset = VideoDataset(video_list, labels)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = I3D(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.long())  # Convert targets to Long
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'i3d_activity_recognition.pth')
