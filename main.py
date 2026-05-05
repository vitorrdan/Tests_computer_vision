import cv2
from fall_detector import FallDetector

# ── configuracao ────────────────────────────────────────────────────────────────
VIDEO_SOURCE = 0   # trocar por 0 para webcam
MODEL_PATH   = 'tasks/pose_landmarker_heavy.task'  # modelo usado para marcação de landmarks do mediapipe

# cores por status
CORES = {
    "Normal":               (0, 255, 0), # verde
    "Rastreamento incerto": (128, 128, 128), # cinza
    "Queda Confirmada":     (0, 0, 255), #vermelho
}
COR_SUSPEITA = (0, 165, 255)


def cor_do_status(status: str) -> tuple:
    for chave, cor in CORES.items():
        if chave in status:
            return cor
    return COR_SUSPEITA  # suspeita ou qualquer outro


def desenhar(frame, deteccao: dict):
    """Responsabilidade: apenas visualizacao. Nao toma decisoes."""
    cor = cor_do_status(deteccao["status"])

    # esqueleto
    if deteccao["pontos"]:
        for cx, cy in deteccao["pontos"]:
            cv2.circle(frame, (cx, cy), 5, cor, -1)
        for origem, destino in deteccao["pose_connections"]:
            cv2.line(frame, deteccao["pontos"][origem],
                     deteccao["pontos"][destino], cor, 2)

    # status principal
    cv2.putText(frame, deteccao["status"], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)

    # metricas de debug
    metricas = [
        f"diff_y:     {deteccao['diff_y']:.2f}"      if deteccao["diff_y"]      is not None else "diff_y:     --",
        f"velocidade: {deteccao['velocidade']:.3f}"  if deteccao["velocidade"]  is not None else "velocidade: --",
        f"no chao:    {deteccao['tempo_no_chao']:.1f}s",
        f"visivel:    {deteccao['landmarks_visiveis']}",
    ]
    for i, texto in enumerate(metricas):
        cv2.putText(frame, texto, (10, 60 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    with FallDetector(MODEL_PATH, fps=fps) as detector:
        while True:
            sucesso, frame = cap.read()
            if not sucesso:
                break

            deteccao = detector.process(frame)

            # aqui entra a logica de alerta futura (MQTT, Firebase)
            if deteccao["queda_confirmada"]:
                pass  # disparar_alerta(deteccao)

            desenhar(frame, deteccao)

            cv2.imshow("SafeGuard - Deteccao de Queda", frame)
            if cv2.waitKey(40) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()