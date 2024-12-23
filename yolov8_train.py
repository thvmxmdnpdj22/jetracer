from ultralytics import YOLO

if __name__ == '__main__':

    # 모델 불러오기
    # model = YOLO('models\\TOP&BOTTOM.pt')
    model = YOLO('yolov8n.pt')

    # 모델 학습
    train = model.train(
        data='C:\\Users\\301\\Desktop\\base\\jetracer\\dataset.yaml', 
        epochs=100,  # 학습 반복 횟수
        batch=16,  # 배치 사이즈 설정
        lr0=0.01,  # 초기 학습률 설정
        imgsz=640,  # 이미지 크기 설정
        device=0
    )
