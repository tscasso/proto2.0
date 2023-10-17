import os 
import pandas as pd

data_dir = '/Users/tscasso/Beboo/prototype/data/train/'

data = []

# Parcourir les sous-répertoires correspondant aux émotions
for emotion in os.listdir(data_dir):
    emotion_path = os.path.join(data_dir, emotion)
    if os.path.isdir(emotion_path):
        emotion_label = emotion
        for image_name in os.listdir(emotion_path):
            if image_name.endswith('.png'):
                # Utilisez os.path.abspath pour obtenir le chemin absolu de l'image
                image_path = os.path.abspath(os.path.join(emotion_path, image_name))
                data.append({'image_path': image_path, 'emotion': emotion_label})
                
# Créer un dataframe à partir des données et enregistrer dans un fichier .csv
df = pd.DataFrame(data)

df.to_csv('emotion_label.csv', index=False)
