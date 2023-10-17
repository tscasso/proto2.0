import os 
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

data_dir = 'beboo/data/train'

data = pd.read_csv('/nfs/homes/tscasso/Documents/beboo/emotion_label.csv')

images = []
labels = []

# Parcourir les lignes du DataFrame et charger les images
for index, row in data.iterrows():
    image_path = row['image_path']
    emotion_label = row['emotion']

    # Imprimez le chemin complet de l'image pour vérification
    print(f"Chargement de l'image : {os.path.abspath(image_path)}")

    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        # Redimensionner l'image à la taille souhaitée (par exemple, 48x48)
        image = cv2.resize(image, (48, 48))

        # Normaliser les valeurs des pixels (si nécessaire)
        image = image / 255.0  # Par exemple, diviser par 255 pour obtenir des valeurs entre 0 et 1

        images.append(image)
        labels.append(emotion_label)
    else:
        # En cas d'erreur lors du chargement de l'image
        print(f"Erreur : Impossible de charger l'image {image_path}")
	
# convertir les listes en tableau numpy 

images = np.array(images)
labels = np.array(labels)

# convertir les etiquettes en d'emotions en codage one-hot

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_encoded = to_categorical(labels_encoded)

# diviser les donnees en ensemble d'entrainement et de test 

X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# creer le modele de reconnaissance

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))  # 7 émotions différentes

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64)

# Sauvegarder le modèle
model.save('/nfs/homes/tscasso/Documents/beboo/model/emotion_model.keras')
