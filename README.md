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
## 과업과제분석
### 데이터셋 분석
- 데이터 세트에는 전문가가 생성한 고품질 포토샵 얼굴 이미지가 포함되어 있습니다.
- 이미지는 눈, 코, 입 또는 전체 얼굴로 구분된 서로 다른 얼굴의 합성입니다.
- 실제 사람 이미지인 real: 828장 
- 합성된 사람 이미지 fake: 710장
- 구별 난이도별로
- easy: 3장 mid: 480장hard: 227장
- 데이터는 전부 확인은 하지 않았지만 가로 600px,세로 600px
- 사진은 전부 정면, 회전 없음
``` python
import os
from PIL import Image

folder_path = '/content/drive/MyDrive/hoseoAIDataSet'
all_images_600x600 = True

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            img_path = os.path.join(root, file)
            with Image.open(img_path) as img:
                width, height = img.size
                if width != 600 or height != 600:
                    all_images_600x600 = False
                    print(f"Image {img_path} size is not 600x600: ({width}, {height})")

if all_images_600x600:
    print("All images are 600x600")
```



