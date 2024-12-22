import cv2
import mediapipe as mp

# MediaPipe modüllerini tanımlayalım
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 2) Kamerayı başlat (0 => varsayılan kamera)
cap = cv2.VideoCapture(0)
