import cv2
import mediapipe as mp

# MediaPipe modüllerini tanımlayalım
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# 3) MediaPipe Hands parametrelerini belirleyip 'hands' nesnesi oluşturuyoruz.
with mp_hands.Hands(
    static_image_mode=False,  # Video akışında dinamik el takibi
    max_num_hands=2,         # Aynı anda en fazla 2 el
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    # 3.1) Kamera döngüsü
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Henüz işleyeceğimiz bir şey yok; sadece döngü çalışıyor.
        # Şimdilik ekrana çıktı verelim:
        print("Adım 4 çalışıyor: Kamera döngüsü devam ediyor.")

        # Döngüyü durdurma: 'q' tuşuna basarsak çıkalım
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
