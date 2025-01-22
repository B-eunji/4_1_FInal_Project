import cv2
import dlib
import numpy as np

# Dlib의 얼굴 감지기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# 웹캠 캡처 객체 생성
cap = cv2.VideoCapture(0)

def get_midpoint(p1, p2):
    return ((p1.x + p2.x) // 2, (p1.y + p2.y) // 2)

def calculate_slope(a, b):
    return (b[1] - a[1]) / (b[0] - a[0] + 1e-10)

def calculate_area(landmarks):
    def polygon_area(points):
        area = 0
        for i in range(len(points) - 1):
            area += (points[i].x * points[i + 1].y) - (points[i + 1].x * points[i].y)
        area += (points[-1].x * points[0].y) - (points[0].x * points[-1].y)
        return 0.5 * abs(area)

    left_points = [landmarks.part(i) for i in range(0, 9)]
    right_points = [landmarks.part(i) for i in range(8, 17)]

    left_area = polygon_area(left_points)
    right_area = polygon_area(right_points)

    return left_area, right_area

def is_frontal_face_combined(landmarks):
    # 양쪽 눈 중심 계산
    left_eye_center = get_midpoint(landmarks.part(36), landmarks.part(39))
    right_eye_center = get_midpoint(landmarks.part(42), landmarks.part(45))

    # 입 중심 계산
    mouth_center = get_midpoint(landmarks.part(48), landmarks.part(54))

    # 눈 중심을 이어서 선을 만들고 선의 기울기 계산
    slope_horizontal = calculate_slope(left_eye_center, right_eye_center)

    # 얼굴의 좌우 면적 계산
    left_area, right_area = calculate_area(landmarks)

    # 면적 비율 차이 계산
    area_ratio_diff = abs(left_area - right_area) / (left_area + right_area + 1e-10)

    # 선이 수평에 가까운지 판별 (기울기가 0에 가까운지 확인)
    is_horizontal = abs(slope_horizontal) < 0.5  # 임계값을 0.3으로 설정

    # 얼굴이 정면인지 판별 (수평 및 면적 비율 차이 모두 고려)
    is_frontal = is_horizontal and area_ratio_diff < 0.01  # 면적 비율 차이를 0.2로 설정


    # 설명 생성
    explanation = {
        "is_horizontal": is_horizontal,
        "area_ratio_diff": area_ratio_diff,
        "is_frontal": is_frontal,
        "explanation": f"Horizontal: {'Yes' if is_horizontal else 'No'}, Area Ratio Diff: {area_ratio_diff:.4f}, {'Frontal' if is_frontal else 'Not Frontal'}"
    }

    return explanation

try:
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 전처리
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 감지
        faces = detector(gray)
        for face in faces:
            # 얼굴 특징점 추출
            landmarks = predictor(gray, face)

            # 정면 여부 및 기울기 판별
            explanation = is_frontal_face_combined(landmarks)
            label = "Frontal" if explanation["is_frontal"] else "Not Frontal"

            #어깨 감지
            shoulders = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            #어깨 수평 여부
            if len(shoulders) > 0:
                for (x, y, w, h) in shoulders:
                    # 어깨 중심 계산
                    shoulders_center = (x + w // 2, y + h // 2)
                    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
                    distance = shoulders_center[0] - frame_center[0]
                    is_horizontal = abs(distance) < 20  # 어깨가 중심에 가까운지 여부를 판별
                    break
            else:
                is_horizontal = False

            # 특징점 시각화 및 라벨링
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                # 특징점 시각화
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # 결과 출력
            cv2.putText(frame, label, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, explanation["explanation"], (face.left(), face.top() - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            # cv2.putText(frame, "Shoulders: " + str(is_horizontal), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 화면에 출력
        cv2.imshow('Frame', frame)

        # 종료 조건
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 해제
    cap.release()
    cv2.destroyAllWindows()
