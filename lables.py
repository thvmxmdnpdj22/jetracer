import os

# 기존 라벨링 데이터 파일 경로
input_file = "C:\Users\301\Desktop\base\jetracer\labeling_images2\labels.txt"  # 첫 번째 데이터 파일
input_file_2 = "C:\Users\301\Desktop\base\jetracer\labeling_images3\labels.txt"  # 두 번째 데이터 파일

# YOLO 형식 라벨 파일 저장 디렉토리
output_dir = "C:\Users\301\Desktop\base\jetracer\yolo_labels"
os.makedirs(f"{output_dir}/doll", exist_ok=True)
os.makedirs(f"{output_dir}/goal", exist_ok=True)

# 이미지 크기 (라벨링 시 사용된 이미지 크기와 동일해야 함)
img_width = 640
img_height = 640

# 클래스 ID 매핑 (각 클래스에 고유 ID 할당)
class_mapping = {
    "doll": 0,
    "goal": 1
}

# YOLO 형식 변환 함수
def convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height):
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height

# 파일 읽기 및 변환
def process_file(input_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        file_name = parts[0]  # 이미지 파일 이름
        x1, y1, x2, y2 = map(int, parts[1:5])  # Bounding Box 좌표
        label = parts[5]  # 객체 라벨

        # YOLO 형식 변환
        x_center, y_center, width, height = convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height)
        class_id = class_mapping[label]  # 클래스 ID 가져오기

        # YOLO 형식으로 저장
        output_file = os.path.join(f"{output_dir}/{label}", f"{os.path.splitext(file_name)[0]}.txt")
        with open(output_file, "a") as out_f:
            out_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# 첫 번째 데이터 처리
process_file(input_file)

# 두 번째 데이터 처리
process_file(input_file_2)

print(f"YOLO 형식으로 변환된 파일이 '{output_dir}' 디렉토리에 저장되었습니다.")
