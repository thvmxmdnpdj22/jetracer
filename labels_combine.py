import os

def merge_text_files(folder_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                
                # 텍스트 파일 읽기
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content + '\n')  # 파일 끝에 줄바꿈 추가
                
                print(f"Added: {filename}")

# 사용할 폴더 경로와 출력 파일 설정
folder_path = 'C:\Users\301\Desktop\labelImg\labeled_data\goal_labeled_data'
output_file = 'C:\Users\301\Desktop\base\jetracer\labeling_images3\laels_combine.txt'

merge_text_files(folder_path, output_file)
