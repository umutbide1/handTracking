import cv2
import mediapipe as mp

# MediaPipe modüllerini çağırıyoruz.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Kamerayı (varsayılan 0. kamera) başlatıyoruz.
cap = cv2.VideoCapture(0)

# Hands (El Tespiti) parametreleri ayarlanıyor:
with mp_hands.Hands(
    static_image_mode=False,        # Video akışında sürekli takip
    max_num_hands=2,               # Aynı anda en fazla 2 el
    min_detection_confidence=0.5,   # Tespit eşiği
    min_tracking_confidence=0.5     # Takip eşiği
) as hands:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Görüntüyü yatay eksende (mirror) flip yapıyoruz.
        #    Böylece ekranda gördüğümüz görüntü gerçek hayatta hangi elinizi kaldırdıysanız
        #    o tarafta görünür ve MediaPipe de "Right" / "Left" etiketini doğru verir.
        frame = cv2.flip(frame, 1)

        # 2) OpenCV (BGR) -> MediaPipe (RGB) dönüşümü
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 3) El tespitini gerçekleştiriyoruz.
        results = hands.process(image)

        # 4) Tespit edilen eller varsa (landmarks + handedness)
        if results.multi_hand_landmarks and results.multi_handedness:
            
            # Birden fazla el varsa hepsi için döngü
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                 results.multi_handedness):
                # Elin sağ mı sol mu olduğunun etiketi
                hand_label = handedness.classification[0].label  # 'Left' veya 'Right'
                
                # El üzerindeki landmark'ları ve eklem bağlantılarını çizelim.
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Parmak uc (tip) indeksleri (MediaPipe: 4, 8, 12, 16, 20)
                finger_tips = [4, 8, 12, 16, 20]
                # Parmak PIP (Proximal) eklem indeksleri (2, 6, 10, 14, 18)
                finger_pips = [2, 6, 10, 14, 18]
                
                open_fingers = 0  # Açık parmak sayısı
                
                # Görüntünün boyut bilgilerini alalım.
                h, w, c = frame.shape

                # Her bir parmaktaki tip-pip noktalarını kontrol edelim.
                for tip_idx, pip_idx in zip(finger_tips, finger_pips):
                    tip_landmark = hand_landmarks.landmark[tip_idx]
                    pip_landmark = hand_landmarks.landmark[pip_idx]
                    
                    # x ve y koordinatlarını piksel cinsine çevir.
                    tip_x, tip_y = int(tip_landmark.x * w), int(tip_landmark.y * h)
                    pip_x, pip_y = int(pip_landmark.x * w), int(pip_landmark.y * h)
                    
                    # Başparmak (tip_idx == 4) kontrolü
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
                        # Diğer parmaklar için: tip_y < pip_y => parmak açık
                        if tip_y < pip_y:
                            open_fingers += 1

                # Elin bilek (wrist) noktası (landmark 0)
                wrist_x = int(hand_landmarks.landmark[0].x * w)
                wrist_y = int(hand_landmarks.landmark[0].y * h)
                
                # Ekrana hangi el ve kaç parmağın açık olduğunu yazdıralım.
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
        
        # 5) İşlenmiş görüntüyü ekrana gösteriyoruz.
        cv2.imshow('Hand and Fingers Detection', frame)
        
        # 6) 'q' tuşuna basılırsa çıkış
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
