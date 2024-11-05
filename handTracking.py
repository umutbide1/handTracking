# önce el tespiti frame frame
# önce palm detection sonra handlandmarks yani parmakların tespitini yapacak
# 21 farklı eklem varmış adamlar bunun fizyolojisini çizmiş 


import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0) # farklı kameralar olarabilir ve bu 0 bilgisayarın ilk indexindeki kamera kullanılır

mpHand = mp.solutions.hands # burada mp kütüphanesinden bir obje ürettik
hands = mpHand.Hands()

mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read() # resim şuan BGR çünkü openCV o şekilde okuyor mp ise RGB okuyor
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: 
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)
    
    cv2.imshow("img", img)
    cv2.waitKey(1)