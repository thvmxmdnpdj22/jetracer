import os

def add_prefix_to_images(folder_path):
    for filename in os.listdir(folder_path):
        # 이미지 파일 확장자를 가진 파일만 선택
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # 새로운 파일 이름 생성
            new_filename = 'goal_' + filename
            
            # 파일 경로 변경
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)

            # 파일 이름 변경
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")

# 사용할 폴더 경로 설정
folder_path = 'C:\\Users\\301\\Desktop\\base\\jetracer\\images3'
add_prefix_to_images(folder_path)