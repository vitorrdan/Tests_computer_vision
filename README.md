# SafeGuard Home — Detector de Quedas

## Pré-requisitos

- Python 3.11
- Ambiente virtual criado e ativado
- Dependências instaladas:

```bash
pip install -r requirements.txt
```

- Arquivo do modelo baixado e colocado na pasta `tasks/`:

```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

A estrutura de pastas deve ficar assim:

```
safeguard/
├── main.py
├── fall_detector.py
├── requirements.txt
├── tasks/
│   └── pose_landmarker_heavy.task
└── videos/
    └── video10.mp4  ← seus vídeos de teste aqui
```

---

## Como configurar a fonte de vídeo

Abra o arquivo `main.py` e localize as duas linhas de configuração no topo:

```python
VIDEO_SOURCE = 'videos/video10.mp4'
MODEL_PATH   = 'tasks/pose_landmarker_heavy.task'
```

### Usando a webcam

Troque `VIDEO_SOURCE` pelo índice da câmera — `0` é a câmera integrada do notebook, `1` é uma câmera externa:

```python
VIDEO_SOURCE = 0  # câmera integrada
VIDEO_SOURCE = 1  # câmera externa (ex: TP-Link Tapo via USB)
```

### Usando um vídeo de teste

Coloque o vídeo na pasta `videos/` e aponte o caminho:

```python
VIDEO_SOURCE = 'videos/nome_do_video.mp4'
```

---

## Como rodar

```bash
python main.py
```

Uma janela vai abrir mostrando o vídeo com o esqueleto mapeado e o status da detecção no canto superior esquerdo.

Pressione **Q** para encerrar.

---

## O que aparece na tela

| Cor do esqueleto | Significado |
|------------------|-------------|
| Verde | Normal — nenhuma queda detectada |
| Laranja | Suspeita — possível queda, aguardando confirmação |
| Vermelho | **Queda confirmada** |
| Cinza | Rastreamento incerto — landmarks com baixa confiança |

As métricas no canto esquerdo mostram os valores em tempo real para calibração:

- `diff_y` — diferença vertical entre ombros e quadril. Em pé: ~0.3+. Deitado: próximo de 0
- `velocidade` — variação do centro de massa entre frames. Alta em queda brusca
- `no chao` — há quantos segundos a pessoa está na posição de queda
- `visivel` — se os landmarks principais estão sendo rastreados com confiança

---

## Ajustando os thresholds

Se o detector estiver gerando muitos falsos positivos ou deixando quedas passar, os thresholds ficam no topo do arquivo `fall_detector.py`:

```python
DIFF_Y_NO_CHAO    = 0.13  # diminuir = mais sensível a corpo horizontal
VELOCIDADE_QUEDA  = 0.04  # diminuir = detecta quedas mais lentas
TEMPO_CONFIRMACAO = 2.2   # aumentar = exige mais tempo no chão para confirmar
FRAMES_TOLERANCIA = 20    # aumentar = tolera mais frames sem detecção
```

A forma mais fácil de calibrar é rodar com um vídeo de teste, observar os valores de `diff_y` e `velocidade` na tela em cada situação e ajustar os thresholds de acordo.