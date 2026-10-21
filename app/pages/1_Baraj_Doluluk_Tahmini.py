"""Baraj doluluk tahmini ve erken uyarı sayfası (Sinem)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo kökü: src/ içe aktarılabilsin

import numpy as np
import pandas as pd
import streamlit as st

from src.degerlendir import UFUK, bol, metrikler
from src.ozellikler import model_verisi, modeller
from src.veri import birlesik_veri

KOK = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Baraj Doluluk Tahmini", page_icon="🌊", layout="wide")
st.title("🌊 İstanbul Baraj Doluluk Tahmini ve Erken Uyarı")
st.markdown(
    f"Bugünkü doluluk, son 7/30/90 günün yağışı, su tüketimi ve mevsim bilgisiyle **{UFUK} gün sonraki** "
    "toplam baraj doluluğunu tahmin ediyoruz. Modeller **2011–2022** verisiyle eğitildi, **2023** yılında "
    "hiç görmedikleri veriyle test edildi."
)


@st.cache_data
def veri_getir():
    df = birlesik_veri(str(KOK / "data" / "raw") + "/")
    return df, bol(model_verisi(df))


@st.cache_resource
def modelleri_egit():
    _, p = veri_getir()
    oz = [c for c in p["egitim_tum"].columns if c != "hedef"]
    egitilmis = {}
    for ad, m in modeller().items():
        m.fit(p["egitim_tum"][oz], p["egitim_tum"]["hedef"])
        egitilmis[ad] = m
    return egitilmis, oz


df, p = veri_getir()
egitilmis, oz = modelleri_egit()
test = p["test"]
gercek = test["doluluk"] + test["hedef"]
hedef_tarih = test.index + pd.Timedelta(days=UFUK)

# --- 1. Model başarısı (2023 testi) ---
st.header("1. Model başarısı (2023 testi)")
mevsim = p["egitim_tum"].groupby(p["egitim_tum"].index.dayofyear)["hedef"].mean()
tahminler = {
    "Taban: doluluk değişmez": test["doluluk"].values,
    "Taban: mevsimsel ortalama": test["doluluk"].values + test.index.dayofyear.map(mevsim).fillna(0).values,
}
for ad, m in egitilmis.items():
    tahminler[ad] = test["doluluk"].values + m.predict(test[oz])

tablo = pd.DataFrame({ad: metrikler(gercek, t) for ad, t in tahminler.items()}).T
tablo.columns = ["MAE (yüzde puanı)", "RMSE (yüzde puanı)", "R²"]
st.dataframe(tablo.style.format("{:.2f}").highlight_min(subset=["MAE (yüzde puanı)"], color="#c6efce"),
             width="stretch")
st.caption(
    "MAE: ortalama mutlak hata. RMSE: kök ortalama kare hata (büyük hataları daha çok cezalandırır). "
    "R²: 1'e ne kadar yakınsa o kadar iyi. Tabanlar, modelin gerçekten bir şey öğrenip öğrenmediğini "
    "görmek için basit referans tahminlerdir: bir model 'doluluk değişmez' tahmininden iyi değilse işe yaramaz."
)

lstm_dosya = KOK / "sonuclar" / "karsilastirma_2023.csv"
if lstm_dosya.exists():
    st.subheader("LSTM dahil tüm modeller")
    st.dataframe(pd.read_csv(lstm_dosya, index_col=0).style.format("{:.2f}"), width="stretch")
    st.caption("Bu tablo notebooks/03_beyza_lstm_baraj.ipynb tarafından üretilir (LSTM panelde yeniden eğitilmez, "
               "çünkü eğitimi birkaç dakika sürer).")

secilen = st.multiselect("Grafikte gösterilecek tahminler", list(tahminler),
                         default=["Taban: doluluk değişmez", "Doğrusal Regresyon", "XGBoost"])
grafik = pd.DataFrame({"Gerçek": gercek.values, **{ad: tahminler[ad] for ad in secilen}}, index=hedef_tarih)
grafik.index.name = "Tarih"
st.line_chart(grafik, y_label="Toplam doluluk (%)")

# --- 2. Erken uyarı ---
st.header("2. Erken uyarı")
col1, col2 = st.columns(2)
esik = col1.slider("Uyarı eşiği (%)", 10, 60, 30)
model_adi = col2.selectbox("Model", list(egitilmis), index=0)
tarih = st.date_input(
    "Tahmin yapılan gün (bu günün verisiyle 30 gün sonrası tahmin edilir)",
    value=test.index[-1].date(), min_value=test.index[0].date(), max_value=test.index[-1].date(),
)
satir = test.loc[[pd.Timestamp(tarih)]]
bugun = float(satir["doluluk"].iloc[0])
tahmin = bugun + float(egitilmis[model_adi].predict(satir[oz])[0])
gerceklesen = bugun + float(satir["hedef"].iloc[0])

c1, c2, c3 = st.columns(3)
c1.metric("Bugünkü doluluk", f"%{bugun:.1f}")
c2.metric(f"{UFUK} gün sonrası tahmin", f"%{tahmin:.1f}", f"{tahmin - bugun:+.1f} puan")
c3.metric("Gerçekleşen (kontrol için)", f"%{gerceklesen:.1f}")
if tahmin < esik:
    st.error(f"⚠️ UYARI: {UFUK} gün sonra doluluğun %{esik} eşiğinin altına düşmesi bekleniyor (%{tahmin:.1f}).")
else:
    st.success(f"✅ {UFUK} gün sonra doluluğun %{esik} eşiğinin üzerinde kalması bekleniyor (%{tahmin:.1f}).")

# --- 3. Hangi bilgiler önemli? ---
st.header("3. XGBoost hangi bilgilere bakıyor?")
onem = pd.Series(egitilmis["XGBoost"].feature_importances_, index=oz).sort_values(ascending=False)
st.bar_chart(onem, horizontal=True, x_label="Önem")

st.caption("Veri: İBB Açık Veri Portalı (günlük baraj doluluk, yağış ve İstanbul günlük su tüketimi).")
