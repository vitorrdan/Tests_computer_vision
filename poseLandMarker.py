import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

#Conexoes entre os 33 landmarks do corpo
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),         # rosto esquerdo
    (0,4),(4,5),(5,6),(6,8),         # rosto direito
    (11,12),                          # ombros
    (11,13),(13,15),                  # braco esquerdo
    (12,14),(14,16),                  # braco direito
    (11,23),(12,24),(23,24),          # tronco
    (23,25),(25,27),(27,29),(29,31),  # perna esquerda
    (24,26),(26,28),(28,30),(30,32),  # perna direita   
]

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.IMAGE,
    min_pose_detection_confidence=0.5,  
)

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        sucess, frame = cap.read()
        if not sucess:
            break
        
        h, w, c = frame.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=frame) #frames que o media pipe entende
        result = landmarker.detect(mp_image)
        '''
        PoseLandmarkerResult
        └── pose_landmarks        ← lista de pessoas detectadas
            └── [0]               ← primeira pessoa (lista de 33 landmarks)
                └── [0]           ← landmark 0 (nariz)
                    ├── x
                    ├── y
                    ├── z
                    └── visibility
                └── [1]           ← landmark 1
                ...
                └── [32]          ← landmark 32
        '''
        if result.pose_landmarks: #se conseguiu identificar pessoa
            for pose in result.pose_landmarks: #itera sob as 33 landmarks dela
                
                pontos = []
                for lm in pose:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pontos.append((cx,cy))
                    cv2.circle(frame, (cx,cy), 5, (0,255,0), -1)
                    
                for origem, destino in POSE_CONNECTIONS:
                    cv2.line(frame, pontos[origem], pontos[destino], (255,255,255), 2)
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
                       
                 
cap.release()
cv2.destroyAllWindows                        