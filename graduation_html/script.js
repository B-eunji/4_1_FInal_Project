const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const context = overlay.getContext('2d');
let isFrontal = false;
let timer;

async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true
        });
        video.srcObject = stream;
        console.log('Camera setup complete.');
    } catch (error) {
        console.error('Error accessing camera:', error);
    }
}

async function loadModels() {
    try {
        await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
        await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
        await faceapi.nets.faceRecognitionNet.loadFromUri('/models');
        console.log('Models loaded successfully.');
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

function getMidpoint(p1, p2) {
    return {
        x: (p1.x + p2.x) / 2,
        y: (p1.y + p2.y) / 2
    };
}

function calculateSlope(a, b) {
    return (b.y - a.y) / (b.x - a.x + 1e-10);
}

function calculateAreaRatio(landmarks) {
    const leftPoints = landmarks.slice(0, 8);
    const rightPoints = landmarks.slice(8, 16);
    
    const leftArea = leftPoints.reduce((area, point, i, arr) => {
        if (i < arr.length - 1) {
            area += Math.abs(point.x * arr[i + 1].y - arr[i + 1].x * point.y);
        }
        return area;
    }, 0);

    const rightArea = rightPoints.reduce((area, point, i, arr) => {
        if (i < arr.length - 1) {
            area += Math.abs(point.x * arr[i + 1].y - arr[i + 1].x * point.y);
        }
        return area;
    }, 0);

    const totalArea = leftArea + rightArea;
    return {
        leftRatio: leftArea / totalArea,
        rightRatio: rightArea / totalArea
    };
}

function isFrontalFace(landmarks) {
    const leftEyeCenter = getMidpoint(landmarks[36], landmarks[39]);
    const rightEyeCenter = getMidpoint(landmarks[42], landmarks[45]);
    const slopeHorizontal = calculateSlope(leftEyeCenter, rightEyeCenter);
    const { leftRatio, rightRatio } = calculateAreaRatio(landmarks);
    const areaRatioDiff = Math.abs(leftRatio - rightRatio);
    const isHorizontal = Math.abs(slopeHorizontal) < 0.15;
    const isFrontal = isHorizontal && areaRatioDiff < 0.2;

    const tilt = leftRatio > rightRatio ? "Left" : (leftRatio < rightRatio ? "Right" : "None");

    return {
        isHorizontal,
        areaRatioDiff,
        isFrontal,
        tilt,
        explanation: `Horizontal: ${isHorizontal ? "Yes" : "No"}, Area Ratio Diff: ${areaRatioDiff.toFixed(4)}, ${isFrontal ? "Frontal" : "Not Frontal"}, Tilt: ${tilt}`
    };
}

async function detectFace() {
    const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();
    context.clearRect(0, 0, overlay.width, overlay.height);
    faceapi.draw.drawDetections(overlay, detections);
    faceapi.draw.drawFaceLandmarks(overlay, detections);

    detections.forEach(detection => {
        const landmarks = detection.landmarks.positions;
        const explanation = isFrontalFace(landmarks);
        const label = explanation.isFrontal ? "Frontal" : "Not Frontal";

        landmarks.forEach(point => {
            context.beginPath();
            context.arc(point.x, point.y, 5, 0, 2 * Math.PI);
            context.fillStyle = "green";
            context.fill();
        });

        context.fillStyle = "red";
        context.font = "16px Arial";
        context.fillText(label, detection.detection.box.x, detection.detection.box.y - 10);
        context.fillText(explanation.explanation, detection.detection.box.x, detection.detection.box.y - 30);

        if (explanation.isFrontal) {
            if (!isFrontal) {
                isFrontal = true;
                startTimer();
            }
        } else {
            if (isFrontal) {
                isFrontal = false;
                clearTimeout(timer);
            }
        }
    });
}

function startTimer() {
    timer = setTimeout(() => {
        captureImage();
    }, 3000); // 3초 타이머
}

function captureImage() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // 이미지 데이터를 처리하는 방법 예시 (이미지 다운로드)
    const dataURL = canvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.href = dataURL;
    link.download = 'captured_image.png';
    link.click();
}

document.getElementById('startButton').addEventListener('click', async () => {
    console.log('Start button clicked.');
    await loadModels();
    await setupCamera();
    document.querySelector('.hidden-elements').style.display = 'block'; // Show the hidden elements
    video.onloadeddata = () => {
        console.log('Video data loaded.');
        setInterval(detectFace, 100);
    };
});
