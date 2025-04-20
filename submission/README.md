Berikut README yang bisa langsung kamu **copas dan sesuaikan** sesuai struktur dan hasil proyek kamu:

---

# Food Image Classification with VGG16

Proyek ini melakukan klasifikasi gambar makanan ke dalam 3 kelas: **Sandwich**, **Fries**, dan **Donut** menggunakan Transfer Learning dengan VGG16.

## Struktur Dataset

Dataset diambil dari folder:

```
/content/dataset/Food Classification dataset/
├── Sandwich/
├── Fries/
└── Donut/
```

Data kemudian digabung, dan dibagi ulang menjadi:

- `train/`
- `val/`
- `test/`

## Arsitektur Model

- **Base Model**: VGG16 (tanpa top layer, pretrained dari ImageNet)
- **Ukuran Input**: 150x150 piksel
- **Layer Tambahan**: Beberapa Dense dan Dropout di atas VGG16
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Augmentasi**: Diterapkan hanya pada data training
- **Callback**: ModelCheckpoint dan EarlyStopping

## Augmentasi

Augmentasi dilakukan menggunakan `ImageDataGenerator` untuk memperbanyak variasi pada data training, termasuk rotasi, zoom, shift, dan flip horizontal.

## Training

Model dilatih menggunakan TensorFlow/Keras dengan split:

- 70% train
- 15% validation
- 15% test

## Evaluasi

Evaluasi model dilakukan menggunakan confusion matrix, classification report, dan visualisasi distribusi prediksi.

## Export Model

Model disimpan dalam 3 format:

1. **SavedModel**: `/content/submission/saved_model`
2. **TensorFlow Lite**: `/content/submission/tflite/model.tflite`
   - Label disimpan dalam `label.txt`
3. **TensorFlow.js**: `/content/submission/tfjs_model/`
4. **Keras Format**: `/content/submission/model.keras`

## Inferensi Sample

```python
from tensorflow.keras.preprocessing import image
import numpy as np

sample_path = test_generator.filepaths[0]
img = image.load_img(sample_path, target_size=img_size)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

pred = model.predict(img_array)
pred_class = labels[np.argmax(pred)]
print(f"Sample Prediction: {pred_class}")
```

## Hasil

- Model berhasil memprediksi gambar dengan cukup akurat pada ketiga kelas.
- Akurasi yang dicapai setelah training dan augmentasi cukup tinggi dan stabil.

---
