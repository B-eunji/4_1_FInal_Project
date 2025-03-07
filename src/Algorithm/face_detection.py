from fastapi import FastAPI, File, UploadFile
import cv2
import dlib
import numpy as np
import uvicorn

app = FastAPI()

# Dlib의 얼굴 감지기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


#두 눈의 중점 반환
def get_midpoint(p1, p2):
    return ((p1.x + p2.x) // 2, (p1.y + p2.y) // 2)

#신발끈 공식을 이용한 얼굴 좌우 면적 계산 
def calculate_area(landmarks):
    def polygon_area(points):
        area = 0
        for i in range(len(points) - 1):
            area += (points[i].x * points[i + 1].y) - (points[i + 1].x * points[i].y)
        return 0.5 * abs(area)
    # 얼굴 왼쪽 & 오른쪽 영역 포인트 추출
    left_points = [landmarks.part(i) for i in range(0, 9)]
    right_points = [landmarks.part(i) for i in range(8, 17)]

    left_area = polygon_area(left_points)
    right_area = polygon_area(right_points)

    return left_area, right_area

# 두 점 연결하는 선의 기울기 계산
def calculate_slope(a, b):
    return (b[1] - a[1]) / (b[0] - a[0] + 1e-10)

@app.post("/")
#얼굴 정면 여부 및 기울기 판별 API
async def detect_face(file: UploadFile = File(...)):
    try:
        #이미지 로드 & 그레이스케일 변환
        img_np = np.frombuffer(file.file.read(), np.unit8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLER)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = detector(gray)
        if not faces:
            return {"error": "No face detected"}, 400
        for face in faces:
            #얼굴 랜드마크 검출
            landmarks = predictor(gray, face)
            
            #신발끈 공식 적용하여 좌우 면적 계산(수평 여부)
            left_area, right_area = calculate_area(landmarks)
            area_ratio_diff = abs(left_area - right_area) / (left_area + right_area + 1e-10)
            is_frontal = area_ratio_diff < 0.01
            
            #기울기 계산(눈 중심 기준)
            left_eye_center = get_midpoint(landmarks.part(36), landmarks.part(39))
            right_eye_center = get_midpoint(landmarks.part(42), landmarks.part(45))
            slope_horizontal = calculate_slope(left_eye_center, right_eye_center)
            
            #얼굴 기울기 판별(정면)
            tilt = None
            if slope_horizontal > 0.05:
                tile = "Left"
            elif slope_horizontal < -0.05:
                tilt = "Right"
                    
            return {
                "isFrontal": is_frontal,
                "area_ratio_diff": area_ratio_diff,
                "tile": tile,
                "slope_horizontal": slope_horizontal
            }
            
    except Exception as e:
        return{"error": str(e)}, 500

#FASTAPI 서버 실행 (로컬 테스트용)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)