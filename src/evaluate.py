import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle

def plot_history(history):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Eğitim Doğruluğu')
    plt.plot(epochs, val_acc, 'b', label='Doğrulama Doğruluğu')
    plt.title('Eğitim ve Doğrulama Doğruluğu')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Eğitim Kaybı')
    plt.plot(epochs, val_loss, 'b', label='Doğrulama Kaybı')
    plt.title('Eğitim ve Doğrulama Kaybı')
    plt.legend()
    plt.show()

# Modeli yükle
model = load_model('models/cnn_model.keras')

# Eğitimi yükle
with open('models/history.pkl', 'rb') as f:
    history = pickle.load(f)

plot_history(history)
