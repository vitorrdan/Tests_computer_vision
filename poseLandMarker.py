import cv2
import mediapipe as mp

# Atalhos para classes e enums da API Tasks do MediaPipe
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Conexoes entre os 33 landmarks do corpo
# Cada tupla liga dois pontos para desenhar o esqueleto na tela
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

# Configuracao do modelo de pose
# - model_asset_path: arquivo do modelo .task
# - running_mode: processa cada frame como uma imagem isolada
# - min_pose_detection_confidence: confianca minima para aceitar a deteccao
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='tasks/pose_landmarker_lite.task'),#modelo leve
    running_mode=VisionRunningMode.IMAGE,
    min_pose_detection_confidence=0.5,  
)

# Abre a webcam padrao do computador
cap = cv2.VideoCapture(0)

# Cria o detector e garante que os recursos sejam liberados ao sair
with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        # Lê um frame da camera
        success, frame = cap.read()
        if not success:
            break
        
        # Dimensoes do frame para converter coordenadas normalizadas em pixels
        h, w, c = frame.shape
        
        # Converte o frame do OpenCV para o formato esperado pelo MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Executa a deteccao de pose no frame atual
        result = landmarker.detect(mp_image)
        '''
        result recebe:
        
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
        # Se encontrou uma ou mais pessoas, desenha os landmarks
        if result.pose_landmarks:
            for pose in result.pose_landmarks:
                # Guarda os pontos convertidos para coordenadas de pixel
                pontos = []
                for lm in pose:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pontos.append((cx,cy))
                    cv2.circle(frame, (cx,cy), 5, (0,255,0), -1)
                    
                # Desenha as conexoes entre os pontos do corpo
                for origem, destino in POSE_CONNECTIONS:
                    cv2.line(frame, pontos[origem], pontos[destino], (255,255,255), 2)

        # Exibe o frame com os pontos e linhas desenhados
        cv2.imshow("Camera", frame)

        # Pressione q para encerrar a execucao
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
                       
                 
cap.release()
cv2.destroyAllWindows()