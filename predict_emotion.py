import cv2
import numpy as np
from keras.models import load_model

# Charger le modèle Keras préalablement enregistré (.keras)
model = load_model('/Users/tscasso/beboo/prototype/model/emotion_model.keras')

# Définir une liste des émotions possibles correspondant aux sorties du modèle
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Créez un détecteur de visage Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Capture vidéo depuis la webcam
cap = cv2.VideoCapture(0)  # 0 correspond à la webcam par défaut, vous pouvez changer cela si nécessaire

# Liste pour stocker les émotions détectées au fil du temps
detected_emotions = []

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convertir l'image en niveaux de gris
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Utilisez le détecteur de visage pour détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Redimensionner et prétraiter la région du visage pour l'émotion
        face_roi = gray_frame[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi / 255.0
    
    # Redimensionner l'image à la taille attendue par le modèle (par exemple, 48x48)
    resized_frame = cv2.resize(gray_frame, (48, 48))
    
    # Normaliser les valeurs des pixels (si nécessaire)
    normalized_frame = resized_frame / 255.0  # Par exemple, diviser par 255 pour obtenir des valeurs entre 0 et 1
    
    # Ajouter une dimension pour l'image (le modèle s'attend à une forme [1, 48, 48, 1])
    image = np.expand_dims(normalized_frame, axis=0)
    
    # Effectuer la prédiction avec le modèle
    predictions = model.predict(image)
    
    # Obtenir l'émotion prédite
    predicted_emotion = emotions[np.argmax(predictions)]
    
    # Ajouter l'émotion prédite à la liste
    detected_emotions.append(predicted_emotion)
    
    # Calculer la moyenne des 5 dernières émotions détectées (ou moins s'il n'y en a pas encore 5)
    if len(detected_emotions) > 10:
        detected_emotions = detected_emotions[-5:]
    
    average_emotion = np.mean([emotions.index(e) for e in detected_emotions])
    
    # Afficher la moyenne des émotions prédites
    cv2.putText(frame, f"Moyenne : {emotions[int(average_emotion)]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Afficher une barre de progression pour la puissance de l'émotion
    cv2.rectangle(frame, (10, 60), (10 + int(average_emotion * 20), 80), (0, 0, 255), -1)
    
    # Encadrer le visage détecté
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Afficher la vidéo en temps réel
    cv2.imshow('Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

