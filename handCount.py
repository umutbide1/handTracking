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
        
        # (1) Görüntüyü flip yap
        frame = cv2.flip(frame, 1)

        # (2) BGR -> RGB dönüşümü
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # (3) El tespiti
        results = hands.process(image)
        
        # (4) Eğer eller tespit edildiyse
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                 results.multi_handedness):
                # Elin sağ/sol etiketi
                hand_label = handedness.classification[0].label  # 'Left' veya 'Right'
                
                # El çizimi
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # (5) Parmak uç (tip) ve PIP indeksleri
                finger_tips = [4, 8, 12, 16, 20]
                finger_pips = [2, 6, 10, 14, 18]
                open_fingers = 0

                # (6) Çerçevenin boyut bilgileri
                h, w, c = frame.shape

                # (7) Her parmak için tip ve pip noktasına bakarak "açık mı?" kontrolü
                for tip_idx, pip_idx in zip(finger_tips, finger_pips):
                    tip_landmark = hand_landmarks.landmark[tip_idx]
                    pip_landmark = hand_landmarks.landmark[pip_idx]
                    
                    # Normalize [0..1] => piksel koordinatları
                    tip_x, tip_y = int(tip_landmark.x * w), int(tip_landmark.y * h)
                    pip_x, pip_y = int(pip_landmark.x * w), int(pip_landmark.y * h)
                    
                    # Başparmak (tip_idx == 4)
                    if tip_idx == 4:
                        # Sağ elde başparmak: tip_x < pip_x => açık
                        # Sol elde başparmak: tip_x > pip_x => açık
                        if hand_label == 'Right':
                            if tip_x < pip_x:
                                open_fingers += 1
                        else:  # 'Left'
                            if tip_x > pip_x:
                                open_fingers += 1
                    else:
                        # Diğer parmaklar: tip_y < pip_y => açık
                        if tip_y < pip_y:
                            open_fingers += 1

                # (8) Bilek (wrist) koordinatı (landmark 0)
                wrist_x = int(hand_landmarks.landmark[0].x * w)
                wrist_y = int(hand_landmarks.landmark[0].y * h)
                
                # (9) Yazıyı ekrana bas ("Right Hand - Open Fingers: 3" gibi)
                text = f'{hand_label} Hand - Open Fingers: {open_fingers}'
                cv2.putText(
                    frame,
                    text,
                    (wrist_x - 50, wrist_y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
        
        # (10) Çerçeveyi ekranda göster
        cv2.imshow('Hand and Fingers Detection', frame)

        # 'q' tuşuna basıldığında döngüden çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
# q ile döngü kırılacaktır
print('proje sonucu')