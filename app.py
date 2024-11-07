import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import KNeighborsClassifierr
from sklearn.model_selection import train_test_split

!wget -q -O- ipv4.icanhazip.com

st.title("Aplikasi Prediksi Stunting Anak")

st.sidebar.header("Masukkan Data Anak")
def user_input_features():
    Umur_bulan = st.sidebar.slider("Umur (bulan)", 0, 60)
    Jenis_Kelamin = st.sidebar.selectbox("Jenis Kelamin", (1,2))
    Tinggi_Badan_cm = st.sidebar.slider("Tinggi Badan (cm)", 0, 100)
    Status_Gizi = st.sidebar.selectbox("Status Gizi", (0,1,2,3,))
