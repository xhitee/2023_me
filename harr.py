import urllib.request
import cv2

# 학습된 모델 파일 경로
cascade_file_path = "haarcascade_frontalface_default.xml"

# 모델 파일 다운로드
url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
urllib.request.urlretrieve(url, cascade_file_path)

# 캠 불러오기
cap = cv2.VideoCapture(1)

# 이미지 저장을 위한 변수 초기화
count = 0

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    # 얼굴 검출을 위한 Harr Cascade 초기화
    face_cascade = cv2.CascadeClassifier(cascade_file_path)

    cv2.imshow('Frame', frame)

    # 얼굴 검출
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 얼굴이 검출되었을 때 이미지 저장
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # 얼굴 영역 추출
            face_roi = frame[y:y+h, x:x+w]

            # 이미지 리사이즈
            face_resized = cv2.resize(face_roi, (112, 112))

            save_path = '/Users/white/Downloads/downloaded - 복사본/NWPU-RESISC45/baek'

            # 이미지 저장
            cv2.imwrite(f'{save_path}/{count}.jpg', face_resized)
            count += 1

    # 얼굴 검출 결과를 화면에 표시
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 화면에 출력
    cv2.imshow('Face Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
