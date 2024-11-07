import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

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
        Jenis_Kelamin = st.selectbox("Jenis Kelamin", 
                                    options=['Laki-laki', 'Perempuan'],
                                    format_func=lambda x: 'Laki-laki' if x == 'Laki-laki' else 'Perempuan')
    
    with col2:
        Tinggi_Badan_cm = st.number_input("Tinggi Badan (cm)", min_value=0.0, max_value=200.0, value=45.0)
        Status_Gizi = st.selectbox("Status Gizi", 
                                 options=['Severely Stunted', 'Stunted', 'Normal', 'Tinggi'],
                                 index=2)
    
    # Konversi input ke format yang sesuai dengan model
    jk_encoded = 1 if Jenis_Kelamin == 'Laki-laki' else 2
    status_gizi_map = {
        'Severely Stunted': 0,
        'Stunted': 1,
        'Normal': 2,
        'Tinggi': 3
    }
    status_gizi_encoded = status_gizi_map[Status_Gizi]
    
    # Buat dictionary data
    data = {
        'Umur (bulan)': Umur_bulan,
        'Jenis Kelamin': jk_encoded,
        'Tinggi Badan (cm)': Tinggi_Badan_cm,
        'Status Gizi': status_gizi_encoded
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
    display_df['Jenis Kelamin'] = display_df['Jenis Kelamin'].map({1: 'Laki-laki', 2: 'Perempuan'})
    display_df['Status Gizi'] = display_df['Status Gizi'].map({
        0: 'Severely Stunted',
        1: 'Stunted',
        2: 'Normal',
        3: 'Tinggi'
    })
    st.write(display_df)
    
    # Tombol untuk prediksi
    if st.button('Prediksi Status Stunting'):
        try:
            # Prediksi
            prediction = model.predict(df_input)
            prediction_proba = model.predict_proba(df_input)
            
            # Mapping hasil prediksi
            status_map = {
                0: 'Severely Stunted',
                1: 'Stunted',
                2: 'Normal',
                3: 'Tinggi'
            }
            
            # Tampilkan hasil
            st.subheader('Hasil Prediksi:')
            st.write(f"Status Stunting: **{status_map[prediction[0]]}**")
            
            # Tampilkan probabilitas
            st.subheader('Probabilitas untuk Setiap Kelas:')
            prob_df = pd.DataFrame(
                prediction_proba,
                columns=['Severely Stunted', 'Stunted', 'Normal', 'Tinggi']
            )
            st.write(prob_df)
            
            # Visualisasi probabilitas dengan bar chart
            st.bar_chart(prob_df.T)
            
        except Exception as e:
            st.error(f"Terjadi kesalahan dalam melakukan prediksi: {str(e)}")
            
    # Informasi tambahan
    with st.expander("Informasi Tambahan"):
        st.write("""
        - Severely Stunted: Sangat Pendek
        - Stunted: Pendek
        - Normal: Normal
        - Tinggi: Tinggi
        
        Aplikasi ini menggunakan model machine learning yang telah dilatih dengan data stunting anak.
        Hasil prediksi bersifat estimasi dan sebaiknya dikonsultasikan dengan tenaga kesehatan profesional.
        """)

if __name__ == '__main__':
    main()
