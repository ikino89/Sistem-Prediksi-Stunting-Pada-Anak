import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the saved model
try:
    model = pickle.load(open('model_klasifikasi.knn', 'rb'))
    scaler = StandardScaler()
except:
    st.error("Model tidak ditemukan. Pastikan file 'model_klasifikasi' ada di direktori yang sama.")

st.title("Aplikasi Prediksi Stunting Anak")

st.sidebar.header("Masukkan Data Anak")

def user_input_features():
    Umur_bulan = st.sidebar.slider("Umur (bulan)", 0, 60)
    Jenis_Kelamin = st.sidebar.selectbox("Jenis Kelamin", (1,2), 
                                       format_func=lambda x: "Laki-laki" if x==1 else "Perempuan")
    Tinggi_Badan_cm = st.sidebar.slider("Tinggi Badan (cm)", 0, 100)
    
    # Membuat dictionary dari input user
    data = {
        'Umur (bulan)': Umur_bulan,
        'Tinggi Badan (cm)': Tinggi_Badan_cm,
        'Jenis Kelamin': Jenis_Kelamin
    }
    
    # Mengubah data menjadi dataframe
    features = pd.DataFrame(data, index=[0])
    return features

# Memanggil fungsi untuk mendapatkan input user
df = user_input_features()

# Menampilkan data yang diinput
st.subheader('Data Input User')
st.write(df)

# Membuat fungsi untuk mengkonversi hasil prediksi ke label yang sesuai
def get_status_gizi(prediction):
    if prediction == 0:
        return "Severely Stunted"
    elif prediction == 1:
        return "Stunted"
    elif prediction == 2:
        return "Normal"
    else:
        return "Tinggi"

# Tombol untuk melakukan prediksi
if st.button('Prediksi Status Stunting'):
    try:
        # Melakukan prediksi
        prediction = model.predict(df)
        
        # Mendapatkan label status gizi
        status_gizi = get_status_gizi(prediction[0])
        
        # Menampilkan hasil dengan format yang lebih menarik
        st.subheader('Hasil Prediksi')
        
        # Menggunakan warna berbeda berdasarkan hasil prediksi
        if prediction[0] <= 1:  # Severely stunted atau stunted
            st.error(f'Status Gizi Anak: {status_gizi}')
        elif prediction[0] == 2:  # Normal
            st.success(f'Status Gizi Anak: {status_gizi}')
        else:  # Tinggi
            st.info(f'Status Gizi Anak: {status_gizi}')
        
        # Menambahkan informasi tambahan
        if prediction[0] <= 1:
            st.warning("""
            Rekomendasi:
            1. Segera konsultasikan dengan dokter atau ahli gizi
            2. Pastikan asupan gizi seimbang
            3. Rutin memantau pertumbuhan anak
            4. Pastikan pemberian ASI atau makanan pendamping yang tepat
            """)
            
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam melakukan prediksi: {str(e)}")

# Menambahkan informasi tambahan
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Keterangan Status Gizi:
- **Severely Stunted**: Sangat Pendek
- **Stunted**: Pendek
- **Normal**: Normal
- **Tinggi**: Tinggi
""")
