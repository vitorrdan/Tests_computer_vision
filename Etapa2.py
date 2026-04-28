# Etapa 2: desenhos na tela
import cv2

cap = cv2.VideoCapture(0)

while True:
    sucess, frame = cap.read()
    
    if not sucess:
        break
    
    #desenha um retangulo
    #params: imagem, ponto_inicial, ponto_final, cor_BGR, espessura
    cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 2)
    
    #escreve texto
    #params: imagem, texto, posicao, fonte, escala, cor, espessura
    cv2.putText(frame, "TESTE", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    
    cv2.imshow("Cam", frame)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    


cap.release()
cv2.destroyAllWindows()