import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
import shutil

# 데이터셋의 기본 경로
dataset = '../../data/oxford-iiit-pet/images'
target_dir = '../../data/oxford-custom'  # 정렬된 파일이 저장될 경로

# 이미지 파일이 있는 디렉토리 검색
for filename in os.listdir(dataset):
    if filename.endswith('.jpg'):  # jpg 파일만 처리
        # 파일 이름에서 마지막 '_' 위치를 찾아 분류 기준으로 사용
        label = filename.rsplit('_', 1)[0]
        # 해당 라벨의 서브디렉토리 경로 생성
        label_dir = os.path.join(target_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)  # 라벨 디렉토리가 없으면 생성
        
        # 파일을 적절한 서브디렉토리로 이동
        src_file = os.path.join(dataset, filename)
        dst_file = os.path.join(label_dir, filename)
        shutil.move(src_file, dst_file)  # 파일 이동

print("Files have been sorted into subdirectories based on their labels.")
