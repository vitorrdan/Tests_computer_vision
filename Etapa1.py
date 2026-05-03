#Etapa 1 OpenCV: Capturar camera, manipular frames
import cv2

cap = cv2.VideoCapture(0)

# loop lê frame por frame
while True:
    sucess, frame = cap.read() #le um frame
    
    if not sucess:
        break
    
    print(frame.shape) #ex: (480, 640, 3)
    
    cv2.imshow("Camera", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()    