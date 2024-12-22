import cv2
import mediapipe as mp

# 1) MediaPipe modüllerini tanımlayalım.
mp_hands = mp.solutions.hands                # El tespiti (Hands) modülü
mp_drawing = mp.solutions.drawing_utils      # Landmark çizim fonksiyonları
mp_drawing_styles = mp.solutions.drawing_styles
