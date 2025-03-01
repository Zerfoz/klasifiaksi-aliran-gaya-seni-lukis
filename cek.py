import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Muat model
model = load_model('keras_model.h5')

# Kompilasi model (jika diperlukan)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Contoh pemuatan data (ganti dengan dataset yang sesuai)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Ubah label menjadi one-hot encoding (ganti sesuai dengan dataset Anda)
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Skala gambar
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Latih ulang model dan simpan riwayat pelatihan
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Plot akurasi dan loss selama epoch
plt.figure(figsize=(12, 4))

# Plot Akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Akurasi Latihan')
plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss Latihan')
plt.plot(history.history['val_loss'], label='Loss Validasi')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
