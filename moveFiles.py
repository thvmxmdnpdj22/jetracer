import os
import shutil

label = "goal"

# 경로 설정
images_source = "images3"  # 원본 이미지 폴더
labels_source = f"yolo_labels/{label}"  # 원본 라벨 폴더
train_images_dest = "train/images"  # 대상 이미지 폴더
train_labels_dest = "train/labels"  # 대상 라벨 폴더

val_images_dest = "val/images"  # 대상 이미지 폴더
val_labels_dest = "val/labels"  # 대상 라벨 폴더

test_images_dest = "test/images"  # 대상 이미지 폴더

# 폴더가 없으면 생성
os.makedirs(train_images_dest, exist_ok=True)
os.makedirs(train_labels_dest, exist_ok=True)

# 이미지와 라벨 파일 매칭 및 이동
def move_and_rename_files(images_source, labels_source, train_images_dest, train_labels_dest, max_files=200):
    val_count = 50
    # 이미지 파일 리스트 가져오기 (최대 max_files개만 가져오기)
    image_files = sorted([f for f in os.listdir(images_source) if f.endswith(('.png', '.jpg', '.jpeg'))])

    labels = sorted([f for f in os.listdir(labels_source) if f.endswith('.txt')])
    label_files = labels[:max_files]
    val_label_files = labels[max_files: max_files+val_count]

    prev_base_name = []

    print(val_label_files.__len__())
    print(val_label_files)

    # 파일 이름 매칭
    count = 0  # 가져온 파일 개수 확인용
    # train에 넣기 201개
    for image_file in image_files:
        # 이미지 파일 이름에서 확장자 제거
        base_name = os.path.splitext(image_file)[0]
        # 라벨 파일 이름도 동일한지 확인
        label_file = f"{base_name}.txt"
        if label_file in label_files:
            # 새로운 파일 이름 설정
            new_image_name = f"{label}_{image_file}"
            new_label_name = f"{label}_{label_file}"

            # 원본 경로
            image_src_path = os.path.join(images_source, image_file)
            label_src_path = os.path.join(labels_source, label_file)

            # 대상 경로
            image_dest_path = os.path.join(train_images_dest, new_image_name)
            label_dest_path = os.path.join(train_labels_dest, new_label_name)

            # 파일 이동
            shutil.copy(image_src_path, image_dest_path)
            shutil.copy(label_src_path, label_dest_path)

            count += 1
            # print(f"Moved: {image_file} -> {new_image_name}")
            # print(f"Moved: {label_file} -> {new_label_name}")

            prev_base_name.append(base_name)

            # 최대 파일 개수에 도달하면 종료
            if count >= max_files:
                break
        else:
            # print(f"No matching label for: {image_file}")
            pass
        
    # val 넣기 51개
    count = 0
    for image_file in image_files:
        # 이미지 파일 이름에서 확장자 제거
        base_name = os.path.splitext(image_file)[0]
        # 라벨 파일 이름도 동일한지 확인
        label_file = f"{base_name}.txt"
        if label_file in val_label_files:
            # 새로운 파일 이름 설정
            new_image_name = f"{label}_{image_file}"
            new_label_name = f"{label}_{label_file}"

            # 원본 경로
            image_src_path = os.path.join(images_source, image_file)
            label_src_path = os.path.join(labels_source, label_file)

            # 대상 경로
            image_dest_path = os.path.join(val_images_dest, new_image_name)
            label_dest_path = os.path.join(val_labels_dest, new_label_name)

            # 파일 이동
            shutil.copy(image_src_path, image_dest_path)
            shutil.copy(label_src_path, label_dest_path)

            count += 1
            # print(f"Moved: {image_file} -> {new_image_name}")
            # print(f"Moved: {label_file} -> {new_label_name}")
            
            prev_base_name.append(base_name)

            # 최대 파일 개수에 도달하면 종료
            if count >= val_count:
                break
        else:
            # print(f"No matching label for: {image_file}")
            pass
    
    # test 넣기 51개
    count = 0
    for image_file in image_files:
        # 이미지 파일 이름에서 확장자 제거
        base_name = os.path.splitext(image_file)[0]
        # 라벨 파일 이름도 동일한지 확
        # 새로운 파일 이름 설정
        if base_name in prev_base_name:
            continue

        new_image_name = f"{label}_{image_file}"

        # 원본 경로
        image_src_path = os.path.join(images_source, image_file)

        # 대상 경로
        image_dest_path = os.path.join(test_images_dest, new_image_name)

        # 파일 이동
        shutil.copy(image_src_path, image_dest_path)

        count += 1
        # print(f"Moved: {image_file} -> {new_image_name}")


# 실행
if __name__ == "__main__":
    move_and_rename_files(images_source, labels_source, train_images_dest, train_labels_dest, max_files=201)
    print("파일 이동 및 이름 변경이 완료되었습니다.")
