import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. SAYFA AYARLARI
st.set_page_config(page_title="İBB Su Tüketim Tahmini", page_icon="💧", layout="wide")
st.title("💧 İstanbul İlçe Bazlı Su Tüketim Tahmin Paneli")
st.markdown(
    "Bu panel, geçmiş tüketim verilerini inceleyerek LSTM derin öğrenme modeli ile su tüketim trendlerini analiz eder."
)


# 2. VERİ VE MODEL YÜKLEME
@st.cache_resource
def modeli_getir():
    return load_model("src/lstm_su_tahmin_modeli.keras")


@st.cache_data
def veriyi_getir():
    return pd.read_csv("data/processed/temizlenmis_gunluk_tuketim.csv")


model = modeli_getir()
df = veriyi_getir()

# 3. YAN MENÜ (SIDEBAR)
st.sidebar.header("⚙️ Kontrol Paneli")
ilceler = df["İlçe"].unique()
varsayilan_index = list(ilceler).index("ADALAR") if "ADALAR" in ilceler else 0
secilen_ilce = st.sidebar.selectbox(
    "Bir İlçe Seçiniz:", ilceler, index=varsayilan_index
)

st.info(
    "⚠️ Not: Modelimiz şu an için sadece ADALAR ilçesinin geçmiş verileriyle eğitilmiştir. Diğer ilçelerdeki tahminler, ADALAR'dan öğrenilen genel patern üzerinden yapılacaktır."
)

df_ilce = df[df["İlçe"] == secilen_ilce].copy()

# 4. VERİ ÖN İŞLEME VE PENCERELEME (Eğitimdeki işlemlerin simülasyonu)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_ilce[["Gunluk_Tuketim_m3"]])


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : (i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


ZAMAN_ADIMI = 30
X, y_gercek_scaled = create_sequences(scaled_data, ZAMAN_ADIMI)

st.subheader(f"📊 {secilen_ilce} - Gerçek vs Tahmin Edilen Tüketim (m³)")

# 5. TAHMİN YAPMA VE EKRANA BASMA
if len(X) > 0:
    with st.spinner("Yapay zeka tahminleri hesaplıyor..."):
        # Modele veriyi verip tahminleri alıyoruz
        y_pred_scaled = model.predict(X)

        # 0-1 arasındaki veriyi tekrar metreküp (m³) cinsine çeviriyoruz
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_gercek = scaler.inverse_transform(y_gercek_scaled.reshape(-1, 1))

        # Grafikte çizdirmek için ortak bir DataFrame oluşturuyoruz
        tahmin_df = pd.DataFrame(
            {"Gerçek Tüketim": y_gercek.flatten(), "LSTM Tahmini": y_pred.flatten()}
        )

        # Sonucu arayüz grafiği olarak ekrana basıyoruz
        st.line_chart(
            tahmin_df, color=["#0000FF", "#FF0000"]
        )  # Gerçek: Mavi, Tahmin: Kırmızı

        # Modelin hata metriklerini hesaplayıp arayüzde estetik kutular (metric) içinde gösteriyoruz
        rmse = np.sqrt(mean_squared_error(y_gercek, y_pred))
        mae = mean_absolute_error(y_gercek, y_pred)

        st.markdown("### Model Performans Metrikleri")
        col1, col2 = st.columns(2)
        col1.metric("Ortalama Mutlak Hata (MAE)", f"{mae:.2f} m³")
        col2.metric("Hata Yön Sapması (RMSE)", f"{rmse:.2f} m³")
else:
    st.warning(
        "Bu ilçe için yeterli veri bulunmuyor (Modelin geçmişe bakması için en az 30 gün gereklidir)."
    )
