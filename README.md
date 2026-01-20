# TrashNet Image Classification (Streamlit Deployment)

Aplikasi ini merupakan implementasi sistem **klasifikasi sampah berbasis citra** menggunakan dataset **TrashNet**. Model yang digunakan adalah **MobileNetV3 Large** yang telah dioptimasi menggunakan **Structured Weight Pruning** dan **Post-Training Quantization (PTQ)**, kemudian dikonversi ke format **TensorFlow Lite (.tflite)** menggunakan **FP16** dan dideploy sebagai aplikasi web menggunakan **Streamlit**.

---

## 🔗 Demo & Repository

- **Repository GitHub TrashNet**  
  https://github.com/igustimadewarditawinaya/TrashNet

- **Aplikasi Streamlit (Public Deployment)**  
  https://trashnet-classification.streamlit.app/

---

## 📌 Fitur Aplikasi

- Upload citra sampah (.jpg, .jpeg, .png)
- Klasifikasi otomatis ke dalam **6 kelas sampah**
- Menampilkan:
  - Hasil prediksi kelas
  - Nilai confidence relatif
  - Distribusi probabilitas seluruh kelas
- Deteksi prediksi ambigu menggunakan **gap-based decision**

---

## 🧠 Model & Metodologi

- **Arsitektur**: MobileNetV3 Large  
- **Optimasi Model**:
  - Structured Pruning
  - Post-Training Quantization (FP16)
- **Framework**: TensorFlow Lite
- **Dataset**: TrashNet
- **Preprocessing**: `preprocess_input` bawaan MobileNetV3

Model dikembangkan dan dilatih secara terpisah, kemudian dikonversi ke format `.tflite` untuk keperluan deployment yang lebih efisien.

---

## 📝 Catatan Penting

- Nilai confidence pada model tidak digunakan sebagai probabilitas absolut, melainkan sebagai ukuran relatif antar kelas.
- Untuk meningkatkan stabilitas keputusan, digunakan **gap-based decision** antar dua probabilitas tertinggi.
- Aplikasi ini ditujukan untuk keperluan penelitian dan demonstrasi sistem klasifikasi citra.

---

**Author**  
I Gusti Made Wardita Winaya
