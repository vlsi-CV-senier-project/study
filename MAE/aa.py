import os
import requests
import tarfile

def download_and_extract(url, extract_to):
    """주어진 URL에서 파일을 다운로드하고, 지정된 위치에 압축을 풉니다."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # 다운로드 받을 파일명 추출
        file_name = url.split('/')[-1]
        temp_tar_path = os.path.join(extract_to, file_name)
        # tar 파일 저장
        with open(temp_tar_path, 'wb') as file:
            file.write(response.raw.read())
        # tar 파일 압축 해제
        with tarfile.open(temp_tar_path) as tar_file:
            tar_file.extractall(path=extract_to)
        # 임시 tar 파일 삭제
        os.remove(temp_tar_path)
        print(f"Downloaded and extracted {file_name}")
    else:
        print(f"Failed to download {url}")

def main():
    # 데이터셋 다운로드 URL
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

    # 저장할 디렉토리 설정
    data_dir = "/workspace/MAE/data"
    os.makedirs(data_dir, exist_ok=True)

    # 이미지와 애너테이션 다운로드 및 압축 해제
    download_and_extract(images_url, data_dir)
    download_and_extract(annotations_url, data_dir)

if __name__ == "__main__":
    main()


