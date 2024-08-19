import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import PIL
import pandas as pd
import cv2
import numpy as np

st.set_page_config(
    page_title="Sistem Deteksi dan Hitung Jumlah Orang Menggunakan YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Sistem Deteksi dan Hitung Jumlah Orang Menggunakan YOLOv8")

st.markdown("""
    Sistem ini digunakan untuk mendeteksi dan menghitung jumlah orang di area pintu stasiun Bekasi.
    """)

model_path = Path('models/best.pt')

@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as ex:
        st.error(f"Tidak dapat memuat model. Periksa path yang ditentukan: {model_path}")
        st.error(ex)
        return None

model = load_model(model_path)

st.sidebar.header("Pilih Media")
source_radio = st.sidebar.radio("Silahkan Pilih Media.. ", ['Gambar'])
source_imgs = []

if source_radio == 'Gambar':
    uploaded_images = st.sidebar.file_uploader(
        "Unggah Gambar...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_images:
        source_imgs.extend(uploaded_images)

if source_imgs and model:
    detect_button = st.sidebar.button("Mulai Deteksi dan Hitung Objek")

    if detect_button:
        for i, source_img in enumerate(source_imgs, start=1):
            st.markdown(f"## Hasil Deteksi dan Hitung Gambar {i}")
            col1, col2 = st.columns(2)

            with col1:
                try:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(uploaded_image, caption="Gambar yang Diunggah", use_column_width=True)
                    img_array = np.array(uploaded_image)

                except Exception as ex:
                    st.error("Terjadi kesalahan saat membuka gambar.")
                    st.error(ex)
                    continue

            with col2:
                try:
                    temp_img_path = Path(f"temp_image_{i}.jpg")
                    cv2.imwrite(str(temp_img_path), cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

                    results = model.predict(source=str(temp_img_path))
                    temp_img_path.unlink()

                    res_plotted = img_array.copy()
                    boxes = results[0].boxes
                    num_people = len(boxes)  

                    
                    for idx, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf.item() * 100
                        
                        cv2.rectangle(res_plotted, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        text = f'ID: {idx+1}'
                        cv2.putText(res_plotted, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                    text_people_count = f'Jumlah Orang: {num_people}'
                    cv2.putText(res_plotted, text_people_count, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    st.image(res_plotted, caption='Gambar Deteksi', use_column_width=True)

                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR))
                    result_img_bytes = buffer.tobytes()

                except Exception as ex:
                    st.write("Terjadi kesalahan saat memproses deteksi.")
                    st.error(ex)

            with st.container():
                with st.expander(f"Hasil Deteksi Gambar {i}", expanded=True):
                    st.write(f"Jumlah Orang Terdeteksi: {num_people}")
                    st.write(f"Rincian Deteksi:")
                    data = []
                    for idx, box in enumerate(boxes, start=1):
                        obj_conf = box.conf.item() * 100
                        data.append({
                            "ID": f"Person {idx}",
                            "Object Confidence": f"{obj_conf:.1f}%"
                        })
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df)

                    st.download_button(
                        label="Unduh Gambar Hasil Deteksi",
                        data=result_img_bytes,
                        file_name=f"detected_image_{i}.jpg",
                        mime="image/jpeg"
                    )
