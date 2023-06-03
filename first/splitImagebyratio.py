import os
import random
import shutil

def split_dataset(real_path, fake_path, test_ratio, testset_path):
    # 테스트 세트를 저장할 디렉토리를 생성합니다.
    os.makedirs(testset_path, exist_ok=True)

    # "real" 서브디렉토리와 "fake" 서브디렉토리를 생성합니다.
    real_test_path = os.path.join(testset_path, "real")
    fake_test_path = os.path.join(testset_path, "fake")
    os.makedirs(real_test_path, exist_ok=True)
    os.makedirs(fake_test_path, exist_ok=True)

    # real 이미지 파일의 경로를 리스트로 수집합니다.
    real_images = []
    for root, dirs, files in os.walk(real_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, file)
                real_images.append(image_path)

    # fake 이미지 파일의 경로를 리스트로 수집합니다.
    fake_images = []
    for root, dirs, files in os.walk(fake_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, file)
                fake_images.append(image_path)

    # real 이미지의 테스트 세트 분할 비율을 계산합니다.
    num_real_test = int(len(real_images) * test_ratio)

    # fake 이미지의 테스트 세트 분할 비율을 계산합니다.
    num_fake_test = int(len(fake_images) * test_ratio)

    # real 이미지의 테스트 세트를 무작위로 선택하고 이동합니다.
    random_real_test = random.sample(real_images, num_real_test)
    for image_path in random_real_test:
        image_name = os.path.basename(image_path)
        target_path = os.path.join(real_test_path, image_name)
        shutil.copy2(image_path, target_path)

    # fake 이미지의 테스트 세트를 무작위로 선택하고 이동합니다.
    random_fake_test = random.sample(fake_images, num_fake_test)
    for image_path in random_fake_test:
        image_name = os.path.basename(image_path)
        target_path = os.path.join(fake_test_path, image_name)
        shutil.copy2(image_path, target_path)



# 사용 예시
real_images_path = r"C:\Users\User\Documents\GitHub\HoseoAIChallenge\dataset\OriginalDataset\AllDataset\train\fake"
fake_images_path = r"C:\Users\User\Documents\GitHub\HoseoAIChallenge\dataset\OriginalDataset\AllDataset\train\real"
testset_ratio = 0.2
testset_path = r"C:\Users\User\Documents\GitHub\HoseoAIChallenge\dataset\OriginalDataset\AllDataset\test"
split_dataset(real_images_path, fake_images_path, testset_ratio, testset_path)
