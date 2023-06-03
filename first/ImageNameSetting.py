import os
import hashlib
from PIL import Image
import shutil

def remove_duplicate_images(original_path, moved_path):
    original_images = {}
    moved_images = {}

    # 원본 이미지의 경로와 해시 값을 딕셔너리에 저장합니다.
    for root, dirs, files in os.walk(original_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, file)
                image_hash = hash_image(image_path)
                original_images[image_hash] = file

    # 이동된 이미지의 경로와 해시 값을 딕셔너리에 저장합니다.
    for root, dirs, files in os.walk(moved_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, file)
                image_hash = hash_image(image_path)
                moved_images[image_hash] = file

    # 중복된 원본 이미지를 삭제합니다.
    for image_hash, image_name in original_images.items():
        if image_hash in moved_images:
            original_image_path = os.path.join(original_path, image_name)
            os.remove(original_image_path)

def hash_image(image_path):
    # 이미지 파일을 열고 해시 값을 생성합니다.
    with open(image_path, 'rb') as f:
        image = Image.open(f)
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
    return image_hash



# 사용 예시
original_directory = r"C:\Users\User\Documents\GitHub\HoseoAIChallenge\dataset\OriginalDataset\AllDataset\real"
moved_directory = r"C:\Users\User\Documents\GitHub\HoseoAIChallenge\dataset\OriginalDataset\nolabel"
remove_duplicate_images(original_directory, moved_directory)
