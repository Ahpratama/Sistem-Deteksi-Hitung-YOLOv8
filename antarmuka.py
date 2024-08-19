# Pengaturan awal halaman
st.set_page_config(
    page_title="Sistem Deteksi dan Hitung Jumlah Orang Menggunakan YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul aplikasi dan deskripsi
st.title("Sistem Deteksi dan Hitung Jumlah Orang Menggunakan YOLOv8")
st.markdown("""
    Sistem ini digunakan untuk mendeteksi dan menghitung jumlah orang di area pintu stasiun Bekasi.
""")

# Sidebar untuk memilih media dan mengunggah gambar
st.sidebar.header("Pilih Media")
source_radio = st.sidebar.radio("Silahkan Pilih Media.. ", ['Gambar'])

if source_radio == 'Gambar':
    uploaded_images = st.sidebar.file_uploader(
        "Unggah Gambar...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)