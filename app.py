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

# Load model
try:
    with open('model_klasifikasi.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model belum tersedia. Pastikan file model_klasifikasi.pkl ada di direktori yang sama.")
    st.stop()

# Fungsi untuk input data
def user_input_features():
    # Buat dua kolom untuk input
    col1, col2 = st.columns(2)
    
    with col1:
        Umur_bulan = st.number_input("Umur (bulan)", min_value=0, max_value=60, value=0)
        Tinggi_Badan_cm = st.number_input("Tinggi Badan (cm)", min_value=0.0, max_value=200.0, value=45.0)
    
    with col2:
        Jenis_Kelamin = st.selectbox("Jenis Kelamin", 
                                    options=['Laki-laki', 'Perempuan'],
                                    format_func=lambda x: 'Laki-laki' if x == 'Laki-laki' else 'Perempuan')
    
    # Konversi input ke format yang sesuai dengan model
    jk_encoded = 1 if Jenis_Kelamin == 'Laki-laki' else 2
    
    # Buat dictionary data dengan urutan yang sama seperti saat training
    data = {
        'Umur (bulan)': [Umur_bulan],
        'Tinggi Badan (cm)': [Tinggi_Badan_cm],
        'Jenis Kelamin': [jk_encoded]
    }
    
    return pd.DataFrame(data)

# Main
def main():
    # Input data
    st.sidebar.header('Input Data Anak')
    input_df = user_input_features()
    
    # Tampilkan data input
    st.subheader('Data yang Dimasukkan:')
    display_df = input_df.copy()
    display_df['Jenis Kelamin'] = display_df['Jenis Kelamin'].map({1: 'Laki-laki', 2: 'Perempuan'})
    st.write(display_df)
    
    # Tombol untuk prediksi
    if st.button('Prediksi Status Stunting'):
        try:
            # Standardisasi input seperti saat training
            scaler = StandardScaler()
            input_scaled = scaler.fit_transform(input_df)
            
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
            status = status_map[prediction[0]]
            st.markdown(f"### Status Stunting: **{status}**")
            
            # Tambahkan interpretasi hasil
            if status == 'Severely Stunted':
                st.warning("Anak tergolong sangat pendek untuk usianya. Segera konsultasikan dengan tenaga kesehatan.")
            elif status == 'Stunted':
                st.warning("Anak tergolong pendek untuk usianya. Konsultasikan dengan tenaga kesehatan.")
            elif status == 'Normal':
                st.success("Tinggi badan anak normal sesuai usianya.")
            else:  # Tinggi
                st.success("Tinggi badan anak di atas rata-rata untuk usianya.")
            
            # Tampilkan probabilitas
            st.subheader('Probabilitas untuk Setiap Kelas:')
            prob_df = pd.DataFrame(
                prediction_proba,
                columns=['Severely Stunted', 'Stunted', 'Normal', 'Tinggi']
            )
            # Format probabilitas sebagai persentase
            prob_df_display = prob_df.applymap(lambda x: f"{x*100:.2f}%")
            st.write(prob_df_display)
            
            # Visualisasi probabilitas dengan bar chart
            st.bar_chart(prob_df.T)
            
        except Exception as e:
            st.error(f"Terjadi kesalahan dalam melakukan prediksi: {str(e)}")
            
    # Informasi tambahan
    with st.expander("Informasi Tambahan"):
        st.write("""
        ### Kategori Status Stunting:
        - **Severely Stunted (Sangat Pendek)**: Tinggi badan sangat kurang dari normal untuk usia anak
        - **Stunted (Pendek)**: Tinggi badan kurang dari normal untuk usia anak
        - **Normal**: Tinggi badan sesuai dengan usia anak
        - **Tinggi**: Tinggi badan di atas rata-rata untuk usia anak
        
        ### Catatan Penting:
        - Aplikasi ini menggunakan model KNN (K-Nearest Neighbors) yang telah dilatih dengan data stunting anak
        - Hasil prediksi bersifat estimasi dan WAJIB dikonsultasikan dengan tenaga kesehatan profesional
        - Faktor-faktor lain seperti genetik, nutrisi, dan riwayat kesehatan juga mempengaruhi pertumbuhan anak
        """)

if __name__ == '__main__':
    main()
