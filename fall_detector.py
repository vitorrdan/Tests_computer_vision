import mediapipe as mp
from collections import deque

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

# ── thresholds ──────────────────────────────────────────────────────────────────
DIFF_Y_NO_CHAO     = 0.13   # diff_y pequeno = ombro e quadril na mesma altura = deitado
VELOCIDADE_QUEDA   = 0.04   # descida brusca — detecta durante a queda, não depois
TEMPO_CONFIRMACAO  = 2.2    # segundos no chao pra confirmar queda
FRAMES_TOLERANCIA  = 20     # frames sem deteccao antes de resetar (~0.6s a 30fps)
VISIBILIDADE_MIN   = 0.5    # ignora landmark se mediapipe nao tem confianca nele


class FallDetector:
    """
    Responsabilidade: receber um frame, processar com MediaPipe
    e retornar o estado atual da deteccao de queda.
    Nao sabe nada sobre câmera, janela ou visualizacao.
    """

    def __init__(self, model_path: str, fps: float = 30.0):
        self.fps = fps
        self._historico_centro_y  = deque(maxlen=10)  # ultimos 10 valores de centro_y
        self._tempo_no_chao       = 0.0               # quantos segundos a pessoa esta no chao
        self._frames_sem_deteccao = 0
        self._queda_confirmada    = False
        self._suspeita_ativa      = False             # flag: viu queda brusca, aguardando confirmacao

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._landmarker = PoseLandMarker.create_from_options(options)

    def process(self, frame) -> dict:
        """
        Recebe um frame BGR do OpenCV e retorna um dicionario com:
            - status:            str  ('Normal', 'Suspeita', 'Queda Confirmada', 'Rastreamento Incerto')
            - queda_confirmada:  bool
            - suspeita_ativa:    bool
            - tempo_no_chao:     float
            - diff_y:            float | None
            - velocidade:        float | None
            - landmarks_visiveis: bool | None
            - pontos:            list[(cx, cy)] | None  — para desenho externo
            - pose_connections:  list[(int, int)]       — para desenho externo
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result   = self._landmarker.detect(mp_image)

        h, w = frame.shape[:2]

        if result.pose_landmarks:
            self._frames_sem_deteccao = 0
            return self._processar_landmarks(result.pose_landmarks[0], w, h)
        else:
            return self._processar_sem_deteccao()

    def _processar_landmarks(self, pose, w: int, h: int) -> dict:
        ombro_esq   = pose[11]
        ombro_dir   = pose[12]
        quadril_esq = pose[23]
        quadril_dir = pose[24]

        landmarks_visiveis = all(
            lm.visibility >= VISIBILIDADE_MIN
            for lm in [ombro_esq, ombro_dir, quadril_esq, quadril_dir]
        )

        # coleta pontos para visualizacao independente do estado
        pontos = [(int(lm.x * w), int(lm.y * h)) for lm in pose]

        if not landmarks_visiveis:
            # landmarks principais invisiveis — mantem estado, nao reseta
            return {
                "status": "Rastreamento incerto",
                "queda_confirmada": self._queda_confirmada,
                "suspeita_ativa": self._suspeita_ativa,
                "tempo_no_chao": self._tempo_no_chao,
                "diff_y": None,
                "velocidade": None,
                "landmarks_visiveis": False,
                "pontos": pontos,
                "pose_connections": POSE_CONNECTIONS,
            }

        # centro de massa vertical
        centro_y = (ombro_esq.y + ombro_dir.y +
                    quadril_esq.y + quadril_dir.y) / 4

        # diferenca vertical entre quadril e ombro
        # em pe: quadril.y > ombro.y entao diff_y e positivo e grande (~0.3+)
        # deitado: valores proximos, diff_y perto de 0
        diff_y = quadril_esq.y - ombro_esq.y

        self._historico_centro_y.append(centro_y)

        # velocidade calculada sobre janela de frames — mais estavel
        velocidade = 0.0
        if len(self._historico_centro_y) >= 5:
            velocidade = self._historico_centro_y[-1] - self._historico_centro_y[-5]  # positivo = descendo

        pessoa_no_chao = diff_y < DIFF_Y_NO_CHAO
        queda_brusca   = velocidade > VELOCIDADE_QUEDA

        # ── logica de deteccao em dois estagios ─────────────────────────────────

        # estagio 1: detecta a transicao de queda (enquanto mediapipe ainda ve a pessoa)
        if queda_brusca and not self._queda_confirmada:
            self._suspeita_ativa = True

        # estagio 2: confirma pelo tempo no chao OU pela suspeita ativa
        if pessoa_no_chao or self._suspeita_ativa:
            self._tempo_no_chao += 1 / self.fps
        else:
            # so reseta se pessoa claramente voltou a posicao normal
            if diff_y > DIFF_Y_NO_CHAO + 0.1:
                self._tempo_no_chao    = 0.0
                self._suspeita_ativa   = False
                self._queda_confirmada = False

        # confirma queda se ficou no chao por tempo suficiente
        if self._tempo_no_chao >= TEMPO_CONFIRMACAO:
            self._queda_confirmada = True

        if self._queda_confirmada:
            status = "Queda Confirmada"
        elif self._suspeita_ativa or pessoa_no_chao:
            status = f"Suspeita... {self._tempo_no_chao:.1f}s"
        else:
            status = "Normal"

        return {
            "status": status,
            "queda_confirmada": self._queda_confirmada,
            "suspeita_ativa": self._suspeita_ativa,
            "tempo_no_chao": self._tempo_no_chao,
            "diff_y": diff_y,
            "velocidade": velocidade,
            "landmarks_visiveis": True,
            "pontos": pontos,
            "pose_connections": POSE_CONNECTIONS,
        }

    def _processar_sem_deteccao(self) -> dict:
        self._frames_sem_deteccao += 1

        if self._frames_sem_deteccao <= FRAMES_TOLERANCIA:
            # mantem estado — pode ser perda momentanea durante a queda
            if self._suspeita_ativa or self._tempo_no_chao > 0:
                self._tempo_no_chao += 1 / self.fps
                if self._tempo_no_chao >= TEMPO_CONFIRMACAO:
                    self._queda_confirmada = True
                status = f"Suspeita... {self._tempo_no_chao:.1f}s"
            else:
                status = "Normal"
        else:
            # perdeu por muito tempo — reseta
            self._tempo_no_chao    = 0.0
            self._suspeita_ativa   = False
            self._queda_confirmada = False
            status = "Normal"

        return {
            "status": status,
            "queda_confirmada": self._queda_confirmada,
            "suspeita_ativa": self._suspeita_ativa,
            "tempo_no_chao": self._tempo_no_chao,
            "diff_y": None,
            "velocidade": None,
            "landmarks_visiveis": None,
            "pontos": None,
            "pose_connections": POSE_CONNECTIONS,
        }

    def close(self):
        self._landmarker.close()

    # suporte ao with
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()