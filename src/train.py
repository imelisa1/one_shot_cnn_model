from data_preparation import prepare_data
from model import create_model

# Veri setini hazırla
x_train, y_train, x_test, y_test = prepare_data()

# Modeli oluştur
model = create_model()

# Modeli eğit
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# Modeli kaydet
model.save('models/cnn_model.keras')  # Modeli .keras formatında kaydediyoruz

# Eğitimi kaydet
import pickle
with open('models/history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
