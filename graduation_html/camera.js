const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const context = overlay.getContext('2d');
const feedbackElement = document.getElementById('feedback');
const arrowElement = document.getElementById('arrow');
const timerElement = document.getElementById('timer');
const capturedImageContainer = document.getElementById('captured-image-container');
const capturedImage = document.getElementById('captured-image');
const faceGuide = document.getElementById('face-guide'); // 얼굴 가이드 요소 참조

video.width = overlay.width;
video.height = overlay.height;

let countdownIntervalId; // 타이머 인터벌 ID
let isFrontal = false;
let timer;

// 새로고침 및 사진 다운로드 버튼 정의
const refreshButton = document.getElementById('refresh-button');
const downloadButton = document.getElementById('download-button');

async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 413 },  // 증명사진의 2배 크기 (413 * 2) 826
                height: { ideal: 531 },  // 증명사진의 2배 크기 (531 * 2) 1062
                facingMode: 'user'}
        });
        video.srcObject = stream;
        console.log('Camera setup complete.');
    } catch (error) {
        console.error('Error accessing camera:', error);
    }
}

async function loadModels() {
    try {
        await faceapi.nets.tinyFaceDetector.loadFromUri('./models');
        await faceapi.nets.faceLandmark68Net.loadFromUri('./models');
        await faceapi.nets.faceRecognitionNet.loadFromUri('./models');
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

// function calculateAreaRatio(landmarks) {
//     const leftPoints = landmarks.slice(0, 8);
//     const rightPoints = landmarks.slice(8, 16);
    
//     const leftArea = leftPoints.reduce((area, point, i, arr) => {
//         if (i < arr.length - 1) {
//             area += Math.abs(point.x * arr[i + 1].y - arr[i + 1].x * point.y);
//         }
//         return area;
//     }, 0);

//     const rightArea = rightPoints.reduce((area, point, i, arr) => {
//         if (i < arr.length - 1) {
//             area += Math.abs(point.x * arr[i + 1].y - arr[i + 1].x * point.y);
//         }
//         return area;
//     }, 0);

//     const totalArea = leftArea + rightArea;
//     return {
//         leftRatio: leftArea / totalArea,
//         rightRatio: rightArea / totalArea
//     };
// }

function isFrontalFace(landmarks) {
    const leftEyeCenter = getMidpoint(landmarks[36], landmarks[39]);
    const rightEyeCenter = getMidpoint(landmarks[42], landmarks[45]);

    const slopeHorizontal = calculateSlope(leftEyeCenter, rightEyeCenter);
    const isHorizontal = Math.abs(slopeHorizontal) < 0.05;

    const tilt = slopeHorizontal > 0.05 ? "Left" : (slopeHorizontal < -0.05 ? "Right" : "None");

    console.log(`Horizontal: ${isHorizontal ? "Yes" : "No"}, Tilt: ${tilt}`);

    return {
        isHorizontal,
        tilt,
        //slopeHorizontal,
        explanation: `Horizontal: ${isHorizontal ? "Yes" : "No"}, Tilt: ${tilt}`
    };
}

function provideFeedback(tilt) {
    console.log(`Providing feedback for tilt: ${tilt}`);
    arrowElement.innerHTML = ''; // 화살표 초기화
    //guideVideoContainer.style.display = 'none'; // 가이드 영상 숨김
    switch (tilt) {
        case "Left":
            feedbackElement.innerHTML = '고개가 삐뚤어졌으니, <br><span class="highlight">고개를 오른쪽으로 조금만 기울여주세요!</span>';
            arrowElement.innerHTML = '➡️➡️➡️➡️➡️'; // 왼쪽 화살표 표시
            //guideVideo.src = 'leftmoving.mp4'; // 왼쪽 가이드 영상 경로 설정
            //guideVideoContainer.style.display = 'block'; // 가이드 영상 표시
            break;
        case "Right":
            feedbackElement.innerHTML = '고개가 삐뚤어졌으니, <br><span class="highlight">고개를 왼쪽으로 조금만 기울여주세요!</span>';
            arrowElement.innerHTML = '⬅️⬅️⬅️⬅️⬅️'; // 오른쪽 화살표 표시
            //guideVideo.src = 'rightmoving.mp4'; // 오른쪽 가이드 영상 경로 설정
            //guideVideoContainer.style.display = 'block'; // 가이드 영상 표시
            break;
        case "None":
            feedbackElement.textContent = "3초 동안만 정면을 유지해주세요!";
            break;
        default:
            feedbackElement.textContent = "3초 동안만 정면을 유지해주세요!";
    }
}

async function detectFace() {
    const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();
    context.clearRect(0, 0, overlay.width, overlay.height);
    context.save();
    context.scale(-1,1);
    context.translate(-overlay.width,0);

    //faceapi.draw.drawDetections(overlay, detections);
    //faceapi.draw.drawFaceLandmarks(overlay, detections);

    context.restore();

    detections.forEach(detection => {
        const landmarks = detection.landmarks.positions.map(point => ({
            x: point.x, // 좌우 반전 적용
            y: point.y
        }));
        const explanation = isFrontalFace(landmarks);
        //const label = explanation.isFrontal ? "Frontal" : "Not Frontal";
    

        landmarks.forEach(point => {
            context.beginPath();
            context.arc(point.x, point.y, 3, 0, 2 * Math.PI);
            context.fillStyle = "red";
            context.fill();
        });

        // context.fillStyle = "red";
        // context.font = "16px Arial";
        // context.fillText(label, overlay.width - detection.detection.box.x, detection.detection.box.y - 10);
        // context.fillText(explanation.explanation, overlay.width - detection.detection.box.x, detection.detection.box.y - 30);

        // context.fillStyle = "red";
        //     context.font = "16px Arial";
        //     context.fillText(`Slope: ${explanation.slopeHorizontal.toFixed(4)}`, 10, 30);

        provideFeedback(explanation.tilt);

        if (explanation.isHorizontal) {
            if (!isFrontal) {
                isFrontal = true;
                startTimer();
            }
        } else {
            if (isFrontal) {
                isFrontal = false;
                clearInterval(countdownIntervalId); // 타이머 중지
                timerElement.textContent = ''; // 타이머 초기화
            }
        }
    });
    context.restore();
}

function startTimer() {
    let countdown = 3;
    if (timerElement) {
        timerElement.textContent = countdown;
    
        countdownIntervalId = setInterval(() => {
            countdown--;
            if (countdown > 0) {
                timerElement.textContent = countdown;
            } else {
                timerElement.textContent = '';
                clearInterval(countdownIntervalId);
                captureImage();
            }
        }, 1000);
    } else {
        console.error('timerElement not found');
    }
}

function captureImage() {
    console.log('Capturing image.');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // 이미지 데이터를 처리 (증명사진 크기로 다운로드)
    const passportPhotoWidth = 413; // 3.5cm in pixels at 300 DPI
    const passportPhotoHeight = 531; // 4.5cm in pixels at 300 DPI

    const scaledCanvas = document.createElement('canvas');
    scaledCanvas.width = passportPhotoWidth;
    scaledCanvas.height = passportPhotoHeight;
    const scaledContext = scaledCanvas.getContext('2d');
    scaledContext.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, passportPhotoWidth, passportPhotoHeight);

    const dataURL = scaledCanvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.href = dataURL;
    link.download = 'photo.png';
    link.click();

    setTimeout(() => {
        showReturnMessage();
    },1000);
}

function showReturnMessage(){
    const message = document.createElement('div');
    message.textContent = "3초 뒤 시작 화면으로 돌아갑니다.";
    message.style.position = 'fixed';
    message.style.top = '50%';
    message.style.left = '50%';
    message.style.transform = 'translate(-50%, -50%)';
    message.style.backgroundColor = 'rgba(0.7, 0, 0, 1.0)';
    message.style.color = 'red';
    message.style.padding = '50px';
    message.style.borderRadius = '10px';
    message.style.textAlign = 'center';
    message.style.zIndex = '1000';
    message.classList.add('blinking')
    document.body.appendChild(message);

    setTimeout(() => {
        window.location.href = 'http://localhost:8000';
    }, 3000);
}

document.addEventListener('DOMContentLoaded', async () => {
    await loadModels();
    await setupCamera();
    video.onloadeddata = () => {
        console.log('Video data loaded.');
        setInterval(detectFace, 100);
    };
});

refreshButton.addEventListener('click', () => {
    window.location.href = 'http://localhost:8000';
});

downloadButton.addEventListener('click', () => {
    captureImage();
});
