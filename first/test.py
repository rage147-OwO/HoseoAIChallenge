import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 하이퍼파라미터 설정
batch_size = 32
num_epochs = 100
learning_rate = 0.001

# 데이터 경로 설정
train_data_path = r"C:\Users\User\Documents\GitHub\HoseoAIChallenge\dataset\train"
test_data_path = r"C:\Users\User\Documents\GitHub\HoseoAIChallenge\dataset\test"

# 모델 가중치 저장 경로
weight_save_path = r"C:\Users\User\Documents\GitHub\HoseoAIChallenge\first"

# 데이터 전처리 및 변환
transform = transforms.Compose([
    transforms.Resize((600, 600)),  # 이미지 크기 조정
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 이미지 정규화
])

# 학습 데이터셋과 테스트 데이터셋 로드
train_dataset = ImageFolder(root=train_data_path, transform=transform)
test_dataset = ImageFolder(root=test_data_path, transform=transform)

# 데이터 로더 생성
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 150 * 150, 64)  # 입력 크기에 맞게 조정
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)  # 클래스 개수에 맞게 출력 크기 조정

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x




# 신1경망 모델 정의
model = Model()  # 모델을 정의해야 함
model = model.to(device)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 및 평가
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        train_loss += loss.item() * images.size(0)
    
    train_accuracy = 100.0 * train_correct / len(train_dataset)
    train_loss /= len(train_dataset)
    
    model.eval()
    test_loss = 0.0
    test_correct = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
            test_loss += loss.item() * images.size(0)
    
    test_accuracy = 100.0 * test_correct / len(test_dataset)
    test_loss /= len(test_dataset)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    # 각 epoch마다 모델 가중치 저장
    torch.save(model.state_dict(), f"{weight_save_path}/model_epoch_{epoch+1}.pth")
