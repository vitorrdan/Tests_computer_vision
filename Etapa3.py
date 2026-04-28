import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# conexoes entre os 21 landmarks da mao
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # polegar
    (0,5),(5,6),(6,7),(7,8),         # indicador
    (0,9),(9,10),(10,11),(11,12),    # medio
    (0,13),(13,14),(14,15),(15,16),  # anelar
    (0,17),(17,18),(18,19),(19,20),  # mindinho
    (5,9),(9,13),(13,17)             # base da mao
]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()
        if not success:
            break

        h, w, c = frame.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = landmarker.detect(mp_image) 

        if result.hand_landmarks:
            for hand in result.hand_landmarks:

                pontos = []
                for lm in hand:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pontos.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                for origem, destino in HAND_CONNECTIONS:
                    cv2.line(frame, pontos[origem], pontos[destino], (255, 255, 255), 2)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()