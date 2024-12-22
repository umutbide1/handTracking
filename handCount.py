import cv2
import mediapipe as mp

# MediaPipe modüllerini tanımlayalım
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Kamerayı başlat
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Görüntüyü yatay eksende çevir (mirror)
        frame = cv2.flip(frame, 1)

        # BGR -> RGB dönüştür
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # El tespiti
        results = hands.process(image)
        
        # 7) Eğer elde edilen sonuçlarda el tespit edildiyse, çizim yapalım
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                 results.multi_handedness):
                
                # Elin sağ/sol olduğunu bulalım
                hand_label = handedness.classification[0].label  # 'Left' veya 'Right'
                
                # El üzerindeki landmark'ları ve eklem bağlantılarını çiziyoruz
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # 8) Her kareyi ekranda gösterelim
        cv2.imshow('Hand Detection', frame)

        # 'q' tuşuna basılırsa çıkalım
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
