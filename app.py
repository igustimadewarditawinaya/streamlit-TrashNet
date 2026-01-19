import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Trash Classification",
    layout="centered"
)

# HEADER UTAMA
st.markdown(
    """
    <h1 style='text-align: center;'>🗑️ Trash Classification System</h1>
    <p style='text-align: center;'>
    Klasifikasi jenis sampah berbasis citra
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# SIDEBAR INFORMASI
st.sidebar.title("Informasi Model")
st.sidebar.markdown("""
**Arsitektur**  
MobileNetV3 Large  

**Optimasi Model**  
- Structured Pruning  
- Post-Training Quantization (FP16)  

**Framework**  
TensorFlow Lite  

**Jumlah Kelas**  
6 Kelas Sampah
""")

st.sidebar.markdown("---")
st.sidebar.info(
    "Aplikasi klasifikasi 6 jenis sampah "
    "berbasis citra menggunakan dataset TrashNet."
)

# LOAD LABEL
@st.cache_resource
def load_labels(label_path="labels.txt"):
    with open(label_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

labels = load_labels()

# LOAD MODEL TFLITE
@st.cache_resource
def load_tflite_model(model_path="pruned-mobilenetv3_large_fp16.tflite"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_SIZE = input_details[0]["shape"][1]

# PREPROCESSING (MobileNetV3 Large)
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((INPUT_SIZE, INPUT_SIZE))
    image = np.array(image).astype(np.float32)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# INFERENSI
def predict(image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    probabilities = tf.nn.softmax(output_data[0]).numpy()

    predicted_index = np.argmax(probabilities)
    confidence = probabilities[predicted_index]

    return labels[predicted_index], confidence, probabilities

# UI INPUT & OUTPUT
st.subheader("📤 Upload Gambar Sampah")

uploaded_file = st.file_uploader(
    "Pilih gambar (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🖼️ Gambar Input")
        st.image(image, width=350)

    with col2:
        st.markdown("### 📊 Hasil Klasifikasi")

        if st.button("🔍 Jalankan Prediksi"):
            with st.spinner("Model sedang melakukan inferensi..."):
                label, confidence, probs = predict(image)

            st.success(f"**Prediksi Kelas:** {label}")
            st.write("Confidence")
            st.progress(float(confidence))
            st.write(f"Nilai Confidence: **{confidence:.4f}**")

            st.markdown("#### Distribusi Probabilitas Kelas")
            for i, class_name in enumerate(labels):
                st.write(f"{class_name}")
                st.progress(float(probs[i]))

# FOOTER
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 12px;'>"
    "Trash Classification System"
    "</p>",
    unsafe_allow_html=True
)
