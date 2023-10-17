import os
os.environ['TF_KERAS'] = '1'
import cv2
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

# Chargement des données à partir du fichier CSV
data = pd.read_csv('/Users/tscasso/Beboo/prototype/src/emotion_label.csv')

# Initialisation des listes pour les images et les étiquettes
images = []
labels = []

# Parcours des lignes du DataFrame et chargement des images
for index, row in data.iterrows():
    image_path = row['image_path']
    emotion_label = row['emotion']

    # Imprimez le chemin complet de l'image pour vérification
    print(f"Chargement de l'image : {os.path.abspath(image_path)}")

    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        # Redimensionner l'image à la taille souhaitée (48x48)
        image = cv2.resize(image, (48, 48))

        # Normaliser les valeurs des pixels (entre 0 et 1)
        image = image / 255.0

        images.append(image)
        labels.append(emotion_label)
    else:
        # En cas d'erreur lors du chargement de l'image
        print(f"Erreur : Impossible de charger l'image {image_path}")

# Convertir les listes en tableaux numpy
images = np.array(images)
labels = np.array(labels)

# Ajouter la dimension du canal aux images
images = images.reshape(images.shape[0], 48, 48, 1)

# Convertir les étiquettes émotionnelles en encodage one-hot
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_encoded = to_categorical(labels_encoded)

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Créer un objet ImageDataGenerator pour l'augmentation des données
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Appliquer l'augmentation des données à l'ensemble d'entraînement
datagen.fit(X_train)

# Définir une fonction d'objectif pour Optuna
def objective(trial):
    # Hyperparamètres à optimiser
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    num_neurons = trial.suggest_int('num_neurons', 64, 512)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
    l2_reg = trial.suggest_loguniform('l2_reg', 1e-6, 1e-2)

    # Créer le modèle avec les hyperparamètres suggérés
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(num_neurons, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dense(7, activation='softmax'))

    # Compiler le modèle avec le taux d'apprentissage suggéré
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # Entraîner le modèle avec la taille de lot suggérée
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), validation_data=(X_test, y_test), epochs=20, verbose=0)

    # Renvoyer la valeur de perte (loss) sur l'ensemble de validation comme objectif
    return history.history['val_loss'][-1]

# Créer et exécuter une étude Optuna pour l'optimisation bayésienne
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)  # Vous pouvez ajuster le nombre d'essais

# Récupérer les meilleurs hyperparamètres trouvés
best_params = study.best_params
print("Meilleurs hyperparamètres trouvés:", best_params)

# Entraîner le modèle final avec les meilleurs hyperparamètres
best_learning_rate = best_params['learning_rate']
best_batch_size = best_params['batch_size']
best_num_neurons = best_params['num_neurons']
best_dropout_rate = best_params['dropout_rate']
best_l2_reg = best_params['l2_reg']

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), kernel_regularizer=l2(best_l2_reg)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(best_dropout_rate))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(best_l2_reg)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(best_dropout_rate))
model.add(Flatten())
model.add(Dense(best_num_neurons, activation='relu', kernel_regularizer=l2(best_l2_reg)))
model.add(Dense(7, activation='softmax'))

    # Compiler le modèle avec les meilleurs hyperparamètres
model.compile(optimizer=Adam(learning_rate=best_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # Entraîner le modèle avec la taille de lot suggérée
history = model.fit(datagen.flow(X_train, y_train, batch_size=best_batch_size), validation_data=(X_test, y_test), epochs=100, verbose=2)

    # Sauvegarder le modèle final
model.save('/Users/tscasso/Beboo/prototype/model/emotion_model.keras')

    # Afficher les courbes d'apprentissage
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()

# Appel de la fonction d'objectif pour obtenir les courbes et sauvegarder le modèle
objective(study.best_trial)
