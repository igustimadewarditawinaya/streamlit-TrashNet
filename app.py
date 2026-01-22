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

CONFIDENCE_THRESHOLD = 0.70

# HEADER
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

# SIDEBAR
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
def load_labels(path="labels.txt"):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels()

# LOAD MODEL TFLITE
@st.cache_resource
def load_interpreter(model_path="pruned-mobilenetv3_large_fp16.tflite"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_interpreter()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_SIZE = input_details[0]["shape"][1]
INPUT_DTYPE = input_details[0]["dtype"]

# PREPROCESSING (IDENTIK TRAINING)
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)

    image = np.array(image, dtype=np.float32)

    # PREPROCESSING RESMI MOBILENETV3
    image = preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image.astype(INPUT_DTYPE)

# INFERENSI
def predict(image: Image.Image):
    input_data = preprocess_image(image)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]

    # Cek apakah output sudah softmax
    if np.max(output) > 1.0 or np.min(output) < 0.0:
        probs = tf.nn.softmax(output).numpy()
    else:
        probs = output

    idx = np.argmax(probs)
    confidence = probs[idx]

    return labels[idx], confidence, probs

# UI
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

            if confidence < CONFIDENCE_THRESHOLD:
                st.error(
                    "❌ Gambar tidak teridentifikasi sebagai sampah.\n\n"
                    "Silakan unggah gambar yang mengandung objek sampah."
                )
            else:
                st.success(f"Prediksi Kelas: **{label}**")
                st.progress(float(confidence))
                st.write(f"Confidence: **{confidence:.4f}**")

                st.markdown("#### Distribusi Probabilitas")
                for i, class_name in enumerate(labels):
                    st.write(class_name)
                    st.progress(float(probs[i]))

# FOOTER
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 12px;'>"
    "Trash Classification System"
    "</p>",
    unsafe_allow_html=True
)
