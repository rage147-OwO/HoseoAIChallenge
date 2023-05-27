import os
import cv2

directory_path = r"C:\Users\User\Documents\GitHub\HoseoAIChallenge\dataset\train\real"

for file in os.listdir(directory_path):
    # 파일 확장자가 이미지 파일인 경우에만 진행
    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(directory_path, file)
        # 이미지 로드
        image = cv2.imread(image_path)
        # 좌우 반전
        flipped_image = cv2.flip(image, 1)
        # 반전된 이미지 저장 (원본 파일명 앞에 'flipped_' 추가)
        flipped_image_path = os.path.join(directory_path, 'flipped_' + file)
        cv2.imwrite(flipped_image_path, flipped_image)
        print(f"Image flipped and saved: {flipped_image_path}")
