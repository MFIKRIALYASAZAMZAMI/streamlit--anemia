import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Membaca model prediksi anemia
anemia_model = joblib.load('Coba_lagi_RF_model.sav')

# Membaca model clustering anemia (hasil PCA dengan 1 komponen utama)
clustering_model = joblib.load('kmeans_model.sav')

# Membaca model PCA
pca_model = joblib.load('pca_model.sav')

# Membaca model scaler
scaler = joblib.load('scaler_model2.sav')

# Judul web
st.title('Prediksi Anemia Pada Anak Usia 6-59 Bulan')

# Input pengguna
Age_Years_input = st.text_input('Input usia anak dalam tahun')
Sex_input = st.selectbox(
    "Pilih jenis kelamin:",
    ('Laki-Laki', 'Perempuan')
)
gender_mapping = {'Laki-Laki': 1, 'Perempuan': 0}
Sex_y = gender_mapping[Sex_input]

# Menampilkan hasil prediksi
st.write('Anda memilih jenis kelamin:', Sex_input)
RBC_count_in_Millions_input = st.text_input('Input nilai Red Blood Cell (RBC) (10^6/μL)')
HGB_Alltitude_Adjusted_input = st.text_input('Input nilai Hemogoblin (g/dL) ')
HCT_input = st.text_input('Input nilai Hematokrit (%) ')
MCV_input = st.text_input('Input nilai Mean Corpuscular Volume (MCV) (fL)')
MCH_input = st.text_input('Input nilai Mean Corpuscular Hemoglobin (MCH) (pg)')
MCHC_input = st.text_input('Input nilai Mean Corpuscular Hemoglobin Concentration (MCHC) (g/dL)')
RDW_input = st.text_input('Input nilai Red Cell Distribution Width (RDW) (%)')

# Validasi input
if Age_Years_input.strip() and Sex_input.strip() and RBC_count_in_Millions_input.strip() and HGB_Alltitude_Adjusted_input.strip() and HCT_input.strip() and MCV_input.strip() and MCH_input.strip() and MCHC_input.strip() and RDW_input.strip():
    Age_Years = float(Age_Years_input)
    Sex = float(Sex_y)
    RBC_count_in_Millions = float(RBC_count_in_Millions_input)
    HGB_Alltitude_Adjusted = float(HGB_Alltitude_Adjusted_input)
    HCT = float(HCT_input)
    MCV = float(MCV_input)
    MCH = float(MCH_input)
    MCHC = float(MCHC_input)
    RDW = float(RDW_input)

    # Code untuk prediksi
    # Membuat tombol untuk prediksi
    if st.button('Test Prediksi Anemia'):
        input_data = np.array([Age_Years, Sex, RBC_count_in_Millions, HGB_Alltitude_Adjusted, HCT, MCV, MCH, MCHC, RDW]).reshape(1, -1)
        anem_prediction = anemia_model.predict(input_data)

        # Menampilkan hasil prediksi
        if anem_prediction[0] == 1:
            anem_diagnosis = 'Anak tidak terkena Anemia'
            st.success(anem_diagnosis)
        else:
            anem_diagnosis = 'Anak terkena Anemia'
            st.error(anem_diagnosis)

            # Melakukan clustering untuk penderita anemia
            # Scaling hanya pada variabel yang digunakan untuk pengklasteran
            clustering_data = np.array([RBC_count_in_Millions, HGB_Alltitude_Adjusted, HCT, MCV, MCH, MCHC, RDW]).reshape(1, -1)
            clustering_data_scaled = scaler.transform(clustering_data)

            # Terapkan PCA pada data yang di-scaling
            clustering_data_pca = pca_model.transform(clustering_data_scaled)

            anemia_severity = clustering_model.predict(clustering_data_pca)
            if anemia_severity[0] == 0:
                severity = 'Rendah'
            else:
                severity = 'Tinggi'
            
            st.write(f'Tingkat keparahan anemia: {severity}')
else:
    st.warning('Mohon lengkapi semua kolom input.')
