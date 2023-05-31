# HoseoAIChallenge
## 2023 HOSEO AI 프로그래밍 대회
### 대회소개
2023 HOSEO AI 프로그래밍 경진대회에 참여하신 것을 환영합니다.
이번 대회를 통해 여러분의 창의적인 소프트웨어 개발활동을 도모하며 인재를 발굴하고 SW역량을 향상시키고자 합니다.
제공되는 데이터셋을 가지고 전처리를 진행하고, 학습 모델을 개발하여 탐지율을 확인하고 효과적인 전처리 및 모델 개발에 대한 발표 평가를 가지게 됩니다.
개발 언어는 C, C++, Java, Python 등 어떤 언어를 사용하셔도 좋으며, 프로그래밍 환경에 대한 제약사항은 없습니다.
단, 부정행위(Data Leakage 등)가 의심될 경우 0점 처리되오며 이 점을 참고해주시기 바랍니다.😊
요구하는 제출자료의 양식에 맞게 작성하여 제출해주시면 됩니다.
우수한 결과를 도출하는 학생에게는 데이터인재양성 프로그램에 참여할 수 있는 가산점이 주어집니다.
좋은 결과가 있으시길 바라며, 많은 참여 부탁드립니다.
### 대회소개
 - 호서대학교 학부생 전체
### 대회일정
- 본교 대학원생 및 학.석사연계과정생 참여 불가
### 신청일정
신청자 접수 및 데이터셋 제공 2023. 5. 2.(화) ~ 2023. 5. 25 (목)
결과 제출물 수시 접수 2023. 5. 2.(화) ~ 2023. 5. 25 (목)
발표 평가 2023. 5. 26 (금)
심사결과 발표 및 수상 대상자 선정 2023. 5. 26 (금)
### 평가기준
 - 발표평가 70% + 탐지율(성능) 30%
### 상금
1등 (1팀)
30만원
2등 (2팀)
각 20만원
3등 (3팀)
각 10만원
## 과정 및 과제
### 데이터셋 분석
- 데이터 세트에는 전문가가 생성한 고품질 포토샵 얼굴 이미지가 포함되어 있습니다.
- 이미지는 눈, 코, 입 또는 전체 얼굴로 구분된 서로 다른 얼굴의 합성입니다.
- 실제 사람 이미지인 real: 828장 
- 합성된 사람 이미지 fake: 710장
- 구별 난이도별로
- easy: 3장 mid: 480장hard: 227장
- 사진은 전부 정면, 회전 없음
- 데이터는 전부 가로 600px,세로 600px
### 목표 모델 Case
1. ResNet: VGG16에 비해 파라미터가 적으므로 많은 레이어를 사용가능, 레이어별 모델 비교
2. VGG16: 
	1. VGG16 Real/Fake :  VGG16모델을 이용하여 real/fake 이미지를 구분하는 이진 분류 모델
	2. VGG16 Easy 데이터 증강 후 학습:
	3. VGG16 Easy 데이터가중치 부여 후 학습





1. 데이터셋을 0.15비율로 나눠 학습, 테스트셋을 나누기
2. 학습셋의 이미지를 좌우 flip시켜 데이터 증강


## 발표자료목차 ##
**제목: 클래스 활성화 맵(CAM) 및 데이터셋 분석을 활용한 얼굴 이미지 분류 개선**

1. 소개
   - 문제의 배경: 합성 및 실제 얼굴 이미지를 사용한 얼굴 이미지 분류
   - 정확한 분류의 중요성과 다양한 응용 분야의 중요성 소개
   
2. 데이터셋 분석
   - 데이터셋 설명: 고품질 포토샵 합성 얼굴 이미지
   - 실제 이미지 및 합성된 이미지 분포: 실제 (828장) 대 합성 (710장)
   - 이미지 특징: 정면, 회전 없음, 크기: 600px x 600px
   
3. 초기 접근법과 도전 과제
   - 학습 및 테스트 분할 접근법과 제한 사항 설명
   - 테스트 세트에서의 과적합 문제
   
4. 개선된 접근법: 학습-테스트-검증 분할
   - 학습-테스트-검증 분할 소개
   - 검증 세트의 장점
   - 평가 지표: 정확도, 정밀도, 재현율, F1 점수
   
5. 클래스 활성화 맵(CAM)을 통한 모델 결정 이해
   - 클래스 활성화 맵(CAM) 소개
   - 모델의 의사 결정과정 시각화
   - 분류에 중요한 영역 해석
   
6. 결과 및 분석
   - 초기 접근법과 개선된 접근법 비교
   - 테스트 세트의 성능 지표: 정확도, 정밀도, 재현율, F1 점수
   - CAM 분석: 모델의 초점 영역에 대한 통찰력
   
7. 결론과 향후 연구
   - 결과 요약
   - 제한 사항 및 향후 개선 가능성
   - 응용 분야 및 추가 연구 방향
   
8. 질의응답 및 토론
   - 관객으로부터의 질문 및 토론 초대
   
9. 참고문헌
   - 관련 논문, 기사 및 자료 인용
  
   




```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 하이퍼파라미터 설정
batch_size = 64
num_epochs = 100
learning_rate = 0.001

# 데이터 경로 설정
train_data_path = r"/content/drive/MyDrive/dataset/train"
test_data_path = r"/content/drive/MyDrive/dataset/test"

# 모델 가중치 저장 경로
weight_save_path = r"/content/drive/MyDrive/dataset/"

# 데이터 전처리 및 변환
transform = transforms.Compose([
    transforms.Resize((600, 600)),  # 이미지 크기 조정
    transforms.RandomHorizontalFlip(),  # 가로 뒤집기
    transforms.RandomRotation(10),  # 랜덤한 각도로 회전
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
        self.dropout = nn.Dropout(0.5)  # 드롭아웃 추가
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
        x = self.dropout(x)  # 드롭아웃 적용
        x = self.fc2(x)
        return x

# 신경망 모델 정의
model = Model()  # 모델을 정의해야 함
model = model.to(device)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 학습률 스케줄링

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
    
    # 학습률 스케줄링 적용
    scheduler.step()

```
Case1


transform = transforms.Compose([
    transforms.Resize((600, 600)),  # 이미지 크기 조정
    transforms.RandomHorizontalFlip(),  # 가로 뒤집기
    transforms.RandomRotation(10),  # 랜덤한 각도로 회전
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 이미지 정규화
])

