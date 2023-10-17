import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from mtcnn import MTCNN  # Importez MTCNN

# Charger le modèle Keras préalablement enregistré (.keras)
emotion_model = load_model('/Users/tscasso/Beboo/prototype/model/emotion_model.keras')

# Définir une liste des émotions possibles correspondant aux sorties du modèle
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Créer un détecteur de visage MTCNN
face_detector = MTCNN()

# Capture vidéo depuis la webcam
cap = cv2.VideoCapture(0)  # Utilisez l'index de votre caméra si différent

# Liste pour stocker les émotions détectées au fil du temps
detected_emotions = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Détecter les visages dans l'image avec MTCNN
    faces = face_detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']

        # Extraire la région du visage
        face_roi = frame[y:y + h, x:x + w]

        # Redimensionner l'image à la taille attendue par le modèle (48x48)
        face_roi = cv2.resize(face_roi, (48, 48))

        # Prétraiter la région du visage
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = face_roi / 255.0
        face_roi = np.reshape(face_roi, (1, 48, 48, 1))

        # Effectuer la prédiction avec le modèle d'émotion
        emotion_prediction = emotion_model.predict(face_roi)
        emotion_label = emotions[np.argmax(emotion_prediction)]

        # Ajouter l'émotion prédite à la liste
        detected_emotions.append(emotion_label)

        # Calculer la moyenne des 5 dernières émotions détectées (ou moins s'il n'y en a pas encore 5)
        if len(detected_emotions) > 10:
            detected_emotions = detected_emotions[-5:]

        average_emotion = np.mean([emotions.index(e) for e in detected_emotions])

        # Dessiner un rectangle autour du visage et afficher l'émotion prédite
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Average: {emotions[int(average_emotion)]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Afficher la vidéo en temps réel
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
