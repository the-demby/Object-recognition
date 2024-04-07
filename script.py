import tensorflow as tf
import numpy as np
import cv2

# charger notre modèle préentrainé 
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# charger l'image contenant l'objet à detecter
image = cv2.imread('image.png')

# prétraitement de notre image
resized = cv2.resize(image, (224, 224))
resized = tf.keras.preprocessing.image.img_to_array(resized)
resized = tf.keras.applications.mobilenet_v2.preprocess_input(resized)

# prédiction de l'objet dans l'image
predictions = model.predict(np.array([resized]))
decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)

# Afficher les resultats de prédiction
for _, label, score in decoded_predictions[0]:
    print(f'Ceci est peut etre un(e) {label} avec une probabilité de {score*100}%')


