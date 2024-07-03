import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Modeli yükleme
model_path = 'models/cnn_model.keras'  # Modelin kaydedildiği yol
model = load_model(model_path)

def predict_image(model, img_path):
    """
    Bu fonksiyon, belirtilen modeli kullanarak bir görüntünün canlı veya cansız olduğunu tahmin eder.
    """
    img = image.load_img(img_path, target_size=(32, 32))  # Görüntüyü yükle ve yeniden boyutlandır
    img_array = image.img_to_array(img)  # Görüntüyü numpy array'e çevir
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Görüntüyü ölçeklendir ve modelin beklediği formata getir

    prediction = model.predict(img_array)  # Tahmin yap
    return 'Canlı' if np.argmax(prediction) == 1 else 'Cansız'  # Tahmin sonucunu döndür

# Test görüntüsü yolu
img_path = 'data/single_image/image.jpg'  # Tahmin yapmak istediğiniz görüntünün yolu
print(predict_image(model, img_path))  # Tahmin sonucunu yazdır
