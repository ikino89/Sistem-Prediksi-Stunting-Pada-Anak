import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Konfigurasi halaman
st.set_page_config(
    page_title="Aplikasi Prediksi Stunting Anak",
    page_icon="👶",
    layout="wide"
)

# Judul aplikasi
st.title("Aplikasi Prediksi Stunting Anak")
st.write("Aplikasi ini membantu memprediksi status stunting pada anak berdasarkan data yang dimasukkan.")

# Load model dan data training
try:
    with open('model_klasifikasi.pkl', 'rb') as file:
        model = pickle.load(file)
    # Load data training untuk scaler
    training_data = pd.read_csv('data_balita.csv')
    # Konversi data kategorik
    training_data['Jenis Kelamin'] = training_data['Jenis Kelamin'].map({'laki-laki': 1, 'perempuan': 2})
    training_data['Status Gizi'] = training_data['Status Gizi'].map({
        'severely stunted': 0,
        'stunted': 1,
        'normal': 2,
        'tinggi': 3
    })
except FileNotFoundError as e:
    st.error(f"File tidak ditemukan: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan: {str(e)}")
    st.stop()

# Inisialisasi dan fit scaler dengan data training
scaler = StandardScaler()
selected_features = ['Umur (bulan)', 'Tinggi Badan (cm)', 'Jenis Kelamin']
scaler.fit(training_data[selected_features])

# Fungsi untuk input data
def user_input_features():
    col1, col2 = st.columns(2)
    
    with col1:
        Umur_bulan = st.number_input("Umur (bulan)", min_value=0, max_value=60, value=0)
        Jenis_Kelamin = st.selectbox("Jenis Kelamin", 
                                    options=['laki-laki', 'perempuan'])
    
    with col2:
        Tinggi_Badan_cm = st.number_input("Tinggi Badan (cm)", min_value=0.0, max_value=200.0, value=45.0)
    
    # Konversi input ke format yang sesuai dengan model
    jk_encoded = 1 if Jenis_Kelamin == 'laki-laki' else 2
    
    # Buat dictionary data
    data = {
        'Umur (bulan)': Umur_bulan,
        'Jenis Kelamin': jk_encoded,
        'Tinggi Badan (cm)': Tinggi_Badan_cm
    }
    
    return pd.DataFrame(data, index=[0])

# Main
def main():
    # Input data
    st.sidebar.header('Input Data Anak')
    df_input = user_input_features()
    
    # Tampilkan data input
    st.subheader('Data yang Dimasukkan:')
    
    # Konversi kembali nilai-nilai untuk ditampilkan
    display_df = df_input.copy()
    display_df['Jenis Kelamin'] = display_df['Jenis Kelamin'].map({1: 'laki-laki', 2: 'perempuan'})
    st.write(display_df)
    
    # Tombol untuk prediksi
    if st.button('Prediksi Status Stunting'):
        try:
            # Standardisasi input menggunakan scaler yang telah di-fit dengan data training
            input_scaled = scaler.transform(df_input[selected_features])
            
            # Prediksi
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            # Mapping hasil prediksi
            status_map = {
                0: 'Severely Stunted',
                1: 'Stunted',
                2: 'Normal',
                3: 'Tinggi'
            }
            
            # Tampilkan hasil
            st.subheader('Hasil Prediksi:')
            status_prediction = status_map[prediction[0]]
            st.write(f"Status Stunting: **{status_prediction}**")
            
            # Tampilkan probabilitas
            st.subheader('Probabilitas untuk Setiap Kelas:')
            prob_df = pd.DataFrame(
                prediction_proba,
                columns=['Severely Stunted', 'Stunted', 'Normal', 'Tinggi']
            )
            st.write(prob_df)
            
            # Visualisasi probabilitas dengan bar chart
            st.bar_chart(prob_df.T)
            
            # Tambahkan rekomendasi berdasarkan hasil prediksi
            st.subheader('Rekomendasi:')
            if prediction[0] in [0, 1]:  # Severely Stunted atau Stunted
                st.warning("""
                Rekomendasi untuk anak dengan status stunting:
                1. Segera konsultasikan dengan dokter atau ahli gizi
                2. Pastikan asupan gizi seimbang
                3. Berikan suplemen sesuai anjuran dokter
                4. Pantau pertumbuhan secara rutin
                """)
            elif prediction[0] == 2:  # Normal
                st.success("""
                Rekomendasi untuk mempertahankan pertumbuhan normal:
                1. Lanjutkan pola makan sehat dan seimbang
                2. Rutin melakukan pemeriksaan pertumbuhan
                3. Pastikan anak mendapat cukup aktivitas fisik
                """)
            else:  # Tinggi
                st.info("""
                Rekomendasi untuk anak dengan tinggi di atas rata-rata:
                1. Pertahankan pola makan sehat
                2. Pastikan asupan gizi tetap seimbang
                3. Lakukan pemeriksaan rutin untuk memantau pertumbuhan
                """)
            
        except Exception as e:
            st.error(f"Terjadi kesalahan dalam melakukan prediksi: {str(e)}")
            
    # Informasi tambahan
    with st.expander("Informasi Tambahan"):
        st.write("""
        ### Kategori Status Stunting:
        - **Severely Stunted**: Sangat Pendek
        - **Stunted**: Pendek
        - **Normal**: Normal
        - **Tinggi**: Tinggi
        
        ### Tentang Model:
        Model ini menggunakan algoritma K-Nearest Neighbors (KNN) yang telah dilatih dengan data stunting anak.
        Hasil prediksi bersifat estimasi dan sebaiknya dikonsultasikan dengan tenaga kesehatan profesional.
        
        ### Faktor yang Mempengaruhi:
        - Umur anak
        - Jenis kelamin
        - Tinggi badan
        
        ### Catatan Penting:
        Hasil prediksi ini hanya sebagai referensi awal dan tidak menggantikan diagnosa medis profesional.
        """)

if __name__ == '__main__':
    main()
