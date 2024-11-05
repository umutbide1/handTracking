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
pTime = 0
cTime = 0
while True:
    success, img = cap.read() # resim şuan BGR çünkü openCV o şekilde okuyor mp ise RGB okuyor
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB) # burada results içerisinde koordinatlar var ve görüntüde el var mı yok mu diye tracking yapılıyor
    print(results.multi_hand_landmarks) # burada none ya da varsa koordinat ekrana basılıyor
    
    if results.multi_hand_landmarks: # none durumundaysa ekrana bir şey basılmamasını sağlayan koşul
        for handLms in results.multi_hand_landmarks: #handLms adında bir değişken oluşturuldu ve koordinatlara nokta koyma işlemi başlatıldı
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS) # noktalar koyuldu ve arasındaki bağlantı sağlandı
            
            for id, lm in enumerate(handLms.landmark): # burada dosyada bulunan hangi nokta hangi ekleme ait kısmına göre id leri alıyoruz
                print(id,lm)
                h ,w ,c = img.shape
                
                cx,cy = int(lm.x*w) , int(lm.y*h)
                
                # bilek mesela sunumda 0 numara olduğu açıktı 
                if id == 4: # 0 bilek id si olduğundan bileği alacak direkt
                    cv2.circle(img, (cx,cy), 9, (255,0,0 ),cv2.FILLED)
                    
    
    # fps hesabı
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime 
    
    cv2.putText(img, "FPS : "+ str(int(fps)), (10,75), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,0),5)
    
    cv2.imshow("img", img)
    cv2.waitKey(1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    