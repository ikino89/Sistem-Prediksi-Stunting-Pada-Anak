import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(page_title="Aplikasi Prediksi Stunting Anak", layout="wide")

# Title and description
st.title("Aplikasi Prediksi Stunting Anak")
st.write("Aplikasi ini membantu memprediksi status stunting pada anak berdasarkan umur, jenis kelamin, dan tinggi badan.")

# Create two columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Data Anak")
    
    # Create form for input
    with st.form("prediction_form"):
        # Input fields
        umur = st.number_input("Umur (bulan)", min_value=0, max_value=60, value=0)
        jenis_kelamin = st.selectbox(
            "Jenis Kelamin",
            options=['laki-laki', 'perempuan'],
            help="Pilih jenis kelamin anak"
        )
        tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=0.0, max_value=200.0, value=0.0)
        
        # Submit button
        submitted = st.form_submit_button("Prediksi Status")

# Load the model and scaler
try:
    with open('model_klasifikasi.knn', 'rb') as file:
        model = pickle.load(file)
    
    # Load the dataset for scaling reference
    df = pd.read_csv('data_balita.csv')
    
    # Prepare the scaler
    scaler = StandardScaler()
    selected_features = ['Umur (bulan)', 'Tinggi Badan (cm)', 'Jenis Kelamin']
    scaler.fit(df[selected_features])
    
    with col2:
        st.subheader("Hasil Prediksi")
        
        if submitted:
            # Prepare input data
            jenis_kelamin_encoded = 1 if jenis_kelamin == 'laki-laki' else 2
            input_data = np.array([[umur, tinggi_badan, jenis_kelamin_encoded]])
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Convert prediction to status
            status_map = {
                0: 'Severely Stunted',
                1: 'Stunted',
                2: 'Normal',
                3: 'Tinggi'
            }
            
            status = status_map[prediction]
            
            # Display result with styling
            st.markdown("### Status Gizi:")
            if prediction in [0, 1]:
                st.error(f"**{status}**")
            elif prediction == 2:
                st.success(f"**{status}**")
            else:
                st.info(f"**{status}**")
            
            # Display input summary
            st.markdown("### Ringkasan Data:")
            st.write(f"- Umur: {umur} bulan")
            st.write(f"- Jenis Kelamin: {jenis_kelamin}")
            st.write(f"- Tinggi Badan: {tinggi_badan} cm")

except Exception as e:
    st.error("Terjadi kesalahan dalam memuat model. Pastikan file model_klasifikasi tersedia.")
    st.exception(e)

# Add information section
st.markdown("---")
st.markdown("### Informasi Tambahan")
st.write("""
- Severely Stunted: Sangat Pendek
- Stunted: Pendek
- Normal: Normal
- Tinggi: Di atas rata-rata
""")

# Footer
st.markdown("---")
st.markdown("Dibuat dengan ❤️ menggunakan Streamlit dan Scikit-learn")
