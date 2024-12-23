import asyncio
import cv2
import numpy as np
import websockets
from ultralytics import YOLO
import math
import json

# YOLO 모델 로드
model = YOLO('C:\\Users\\301\\Desktop\\base\\jetracer\\VERY_BEST.pt')
class_names = model.names
print(f"클래스: {class_names}")

# 기준점
center_x = 320 ###
center_y = 640

# Jetracer 상태
current_position = {"x": center_x, "y": center_y}
last_goal_location = None  # 마지막 탐지된 목표 위치
receive_images = None  # 실시간 수신 이미지

def calculate_distance(point1, point2):
    return math.sqrt((point2["x"] - point1["x"])**2 + (point2["y"] - point1["y"])**2)

# WebSocket으로 Jetracer 제어 명령 전송
async def send_angle(move_angle, throttle):
    uri = "ws://20.30.177.224:5001"
    try:
        async with websockets.connect(uri) as websocket:
            message = json.dumps({"move_angle": move_angle, "throttle": throttle})
            print(f"보내는 명령: {message}")
            await websocket.send(message)
    except Exception as e:
        print(f"WebSocket 오류: {e}")

def process_detections(boxes, labels):
    global last_goal_location
    goal_location = None
    obstacle_location = []

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        box_center_x = round(((x1 + x2) / 2).item(), 5)
        box_center_y = round(((y1 + y2) / 2).item(), 5)

        if label == "goal":
            goal_location = {"x": box_center_x, "y": box_center_y}
            last_goal_location = goal_location
        else:
            distance = calculate_distance(current_position, {"x": box_center_x, "y": box_center_y})
            obstacle_location.append({"location": {"x": box_center_x, "y": box_center_y}, "distance": distance})


    return goal_location, obstacle_location

# 전역 변수로 이동 방향 및 거리 기록
last_movement = {"direction": None, "distance": 0.0}  # direction: 'left' or 'right', distance: 움직인 거리

async def handle_obstacle(goal_location, obstacle_location):
    global last_movement  # 이동 기록을 수정하기 위해 global 선언

    if obstacle_location:
        # 가장 가까운 장애물 선택
        closest_obstacle = min(obstacle_location, key=lambda o: o["distance"])
        
        if goal_location:  # 목적지와 장애물이 모두 있을 때
            if goal_location["x"] > closest_obstacle["location"]["x"]:  # 장애물이 목적지 왼쪽에 있을 때
                print("목적지가 장애물의 오른쪽에 있음. 오른쪽으로 회피.")
                await send_angle(0.3, 0.3)  # 오른쪽으로 회피
                last_movement = {"direction": "right", "distance": 0.3}  # 이동 기록 저장
            else:  # 장애물이 목적지 오른쪽에 있을 때
                print("목적지가 장애물의 왼쪽에 있음. 왼쪽으로 회피.")
                await send_angle(-0.3, 0.3)  # 왼쪽으로 회피
                last_movement = {"direction": "left", "distance": -0.3}  # 이동 기록 저장
            await asyncio.sleep(1)
        else:  # 목적지가 없을 경우, 이동 반경만큼 회귀
            print("목적지가 없음. 이전 이동 기록을 바탕으로 회귀 중...")
            if last_movement["direction"] == "right":  # 이전에 오른쪽으로 이동했으면 왼쪽으로 회귀
                print("오른쪽으로 이동했으므로 왼쪽으로 회귀.")
                await send_angle(-last_movement["distance"], 0.3)
            elif last_movement["direction"] == "left":  # 이전에 왼쪽으로 이동했으면 오른쪽으로 회귀
                print("왼쪽으로 이동했으므로 오른쪽으로 회귀.")
                await send_angle(-last_movement["distance"], 0.3)
            else:
                print("이동 기록이 없음. 정지.")
                await send_angle(0, 0)
            await asyncio.sleep(1)
    else:  # 장애물도 없고 목적지도 없을 때
        if last_goal_location:  # 마지막 목적지가 있으면 회귀
            print("목표물 회귀 중...")
            await adjust_to_goal(last_goal_location)
        else:  # 마지막 목적지도 없으면 정지
            print("장애물 없음, 목표물 상실. 정지.")
            await send_angle(0, 0)



async def adjust_to_goal(goal_location):
    dx = goal_location["x"] - center_x
    dy = center_y - goal_location["y"]
    angle_to_goal = math.atan2(dy, dx)
    target_angle = math.degrees(angle_to_goal) / 90
    print(f"목적지 방향 계산: 각도 {target_angle}")
    await send_angle(target_angle, 0.2)
    await asyncio.sleep(0.5)

async def detection_image():
    global receive_images

    while True:
        if receive_images is None:
            print("이미지를 수신하지 못했습니다.")
        else:
            results = model(receive_images, conf=0.5)
            for result in results:
                boxes = result.boxes.xyxy
                labels = [class_names[int(label)] for label in result.boxes.cls]

                if len(boxes) > 0:
                    goal_location, obstacle_location = process_detections(boxes, labels)
                    await handle_obstacle(goal_location, obstacle_location)
                else:
                    print("탐지된 객체 없음: 정지 상태 유지")
                    await send_angle(0, 0)
                print(f"YOLO 결과: {results}")
        await asyncio.sleep(0.5)


async def receive_image():
    global receive_images
    uri = "ws://20.30.177.224:5000"
    try:
        async with websockets.connect(uri) as websocket:
            while True:
                data = await websocket.recv()
                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    receive_images = img
                    cv2.imshow("Received Image", img)
                    cv2.waitKey(1)
                else:
                    print("이미지 디코딩 실패")
                await asyncio.sleep(0.1)
    except Exception as e:
        print(f"이미지 수신 오류: {e}")

async def main():
    receive_task = asyncio.create_task(receive_image())
    detection_task = asyncio.create_task(detection_image())
    await asyncio.gather(receive_task, detection_task)

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("프로그램이 사용자에 의해 중단되었습니다.")
