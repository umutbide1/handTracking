# FINGER COUNT 

import cv2 
import mediapipe as mp

cap = cv2.VideoCapture(0)  # burada kendi kameramızı kullanıyoruz

# buradan yakalanan değer 640*480 olacak
# görelim 

while True:
    success,img = cap.read() # success kameradan görüntü alma işleminin başarılı olup olmadığını true false döndürür yazdırıladabilir
    mpHand = mp.solutions.han # 21 noktanın izlenmesi ve el izleme modülü 
    hands = mpHand.Hands()
    
    
    
    
    
    
    
    cv2.imshow("img", img)
    cv2.waitKey(1)
    