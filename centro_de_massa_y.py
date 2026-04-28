#OpenCV y cresce pra baixo

import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
PoseLandMarker = mp.tasks.vision.PoseLandmarker
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

with PoseLandMarker.create_from_options(options) as landmarker:
    while True:
        sucess, frame = cap.read()
        if not sucess:
            break
        
        h, w, c = frame.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=frame) #frames que o media pipe entende
        result = landmarker.detect(mp_image)
        
        if result.pose_landmarks:
            for pose in result.pose_landmarks:
                # ombros
                ombro_esq = pose[11]
                ombro_dir = pose[12]
                
                # quadris
                quadril_esq = pose[23]
                quadril_dir = pose[24]

                # calcular posicao vertical do centro de massa
                centro_y = (ombro_esq.y + ombro_dir.y + quadril_esq.y + quadril_dir.y) / 4

                # calcular angulo do tronco
                # se a pessoa esta em pe, ombros ficam bem acima do quadril (y menor)
                # se caiu, ombros e quadril ficam na mesma altura (y parecido)
                diff_y = quadril_esq.y - ombro_esq.y

                print(f"centro_y: {centro_y:.2f}")
                print(f"diff_y ombro-quadril: {diff_y:.2f}")
        
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        
cap.release()
cv2.destroyAllWindows        
        
        
