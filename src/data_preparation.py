import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

def prepare_data():
    # CIFAR-10 veri setini yükleme
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Sınıfları canlı ve cansız olarak etiketleme
    live_classes = [2, 3, 4, 5, 6, 7]  # Examples: bird, cat, deer, dog, frog, horse
    dead_classes = [0, 1, 8, 9]       # Examples: airplane, automobile, ship, truck

    # Etiketleri dönüştürme
    y_train_live_dead = np.array([1 if y in live_classes else 0 for y in y_train])
    y_test_live_dead = np.array([1 if y in live_classes else 0 for y in y_test])

    # Veri setini ön işleme
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train_live_dead)
    y_test = to_categorical(y_test_live_dead)

    return x_train, y_train, x_test, y_test
