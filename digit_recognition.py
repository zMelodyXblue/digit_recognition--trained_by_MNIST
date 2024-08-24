import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

# 加载训练好的模型
net = Net()
net.load_state_dict(torch.load('num_recognition_Net.params'))
net.eval()

# 设置图像预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

st.title('Handwritten Digit Recognition')

uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # 将图像转换为模型输入
    image = image.convert('L')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = net(image)
        _, predicted = torch.max(output, 1)

    st.write(f"Predicted Digit: {predicted.item()}")