import mediapipe as mp
import cv2
import time

BaseOptions = mp.tasks.BaseOptions
PoseLandMarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32),
]

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_heavy.task'),
    running_mode=VisionRunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
)

# --- variaveis de estado ---
centro_y_anterior = None       # centro de massa do frame anterior
tempo_no_chao = 0              # quantos segundos a pessoa esta no chao
queda_confirmada = False

# thresholds — voce vai calibrar esses valores nos testes
VELOCIDADE_QUEDA    = 0.08   # variacao de centro_y entre frames pra considerar queda brusca
DIFF_Y_NO_CHAO      = 0.10   # diff_y pequeno = ombro e quadril na mesma altura = deitado
TEMPO_CONFIRMACAO   = 2.0    # segundos no chao pra confirmar queda

cap = cv2.VideoCapture(0)

with PoseLandMarker.create_from_options(options) as landmarker:
    while True:
        sucesso, frame = cap.read()
        if not sucesso:
            break

        h, w, c = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = landmarker.detect(mp_image)

        status = "Normal"
        cor_status = (0, 255, 0)  # verde

        if result.pose_landmarks:
            for pose in result.pose_landmarks:

                ombro_esq  = pose[11]
                ombro_dir  = pose[12]
                quadril_esq = pose[23]
                quadril_dir = pose[24]

                # centro de massa vertical
                centro_y = (ombro_esq.y + ombro_dir.y + quadril_esq.y + quadril_dir.y) / 4

                # diferenca vertical entre quadril e ombro
                # em pe: quadril.y > ombro.y entao diff_y é positivo e grande (~0.3+)
                # deitado: valores proximos, diff_y perto de 0
                diff_y = quadril_esq.y - ombro_esq.y

                # velocidade de descida — variacao do centro de massa entre frames
                velocidade = 0
                if centro_y_anterior is not None:
                    velocidade = centro_y - centro_y_anterior  # positivo = descendo
                centro_y_anterior = centro_y

                # --- logica de deteccao ---

                # pessoa esta no chao se diff_y for pequeno (corpo horizontal)
                pessoa_no_chao = diff_y < DIFF_Y_NO_CHAO

                if pessoa_no_chao:
                    tempo_no_chao += 1/30  # assume ~30fps
                else:
                    tempo_no_chao = 0
                    queda_confirmada = False

                # confirma queda se ficou no chao por tempo suficiente
                if tempo_no_chao >= TEMPO_CONFIRMACAO:
                    queda_confirmada = True

                # define status visual
                if queda_confirmada:
                    status = "QUEDA CONFIRMADA"
                    cor_status = (0, 0, 255)   # vermelho
                elif pessoa_no_chao:
                    status = f"Suspeita... {tempo_no_chao:.1f}s"
                    cor_status = (0, 165, 255) # laranja

                # desenha esqueleto
                pontos = []
                for lm in pose:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pontos.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 5, cor_status, -1)

                for origem, destino in POSE_CONNECTIONS:
                    cv2.line(frame, pontos[origem], pontos[destino], cor_status, 2)

                # exibe valores na tela pra voce calibrar
                cv2.putText(frame, f"centro_y:  {centro_y:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.putText(frame, f"diff_y:    {diff_y:.2f}", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.putText(frame, f"velocidade:{velocidade:.3f}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.putText(frame, f"no chao:   {tempo_no_chao:.1f}s", (10, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # status principal
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, cor_status, 2)

        cv2.imshow("SafeGuard - Deteccao de Queda", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()