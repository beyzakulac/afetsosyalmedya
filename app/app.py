"""Ana sayfa: proje tanıtımı ve ilçe bazlı su tüketimi analizi (Beyza)."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

KOK = Path(__file__).resolve().parents[1]
# 2023 dosyasında bazı ilçe adları kısaltılmış
AD_DUZELT = {"B.ÇEKMECE": "BÜYÜKÇEKMECE", "K.ÇEKMECE": "KÜÇÜKÇEKMECE", "G.O.PAŞA": "GAZİOSMANPAŞA"}

st.set_page_config(page_title="İstanbul Su Paneli", page_icon="💧", layout="wide")
st.title("💧 İstanbul Baraj Doluluk ve Su Tüketimi Paneli")
st.markdown(
    "**Ana konu:** İstanbul barajlarının 30 gün sonraki doluluk oranını tahmin etmek ve kritik seviye için erken uyarı vermek. "
    "Bunun için soldaki menüden **Baraj Doluluk Tahmini** sayfasını açın.\n\n"
    "Bu sayfa ise **ek analiz** olarak ilçe bazlı yıllık su tüketimini inceler."
)


@st.cache_data
def yillik_tuketim():
    """2011-2022 yıllık ilçe tüketimi (temizlenmiş veriden) + 2023 gerçek değerleri (aylık dosyanın toplamı)."""
    gunluk = pd.read_csv(KOK / "data" / "temizlenmis_gunluk_tuketim.csv", parse_dates=["Tarih"])
    yillik = gunluk.groupby(["İlçe", gunluk["Tarih"].dt.year])["Gunluk_Tuketim_m3"].sum().unstack()
    yillik.index = yillik.index.str.strip()

    aylik = pd.read_excel(KOK / "data" / "raw" / "ilce-baznda-su-tuketim-miktar-2023.xlsx")
    aylik.columns = [str(c).strip() for c in aylik.columns]
    aylik = aylik.dropna(subset=["İLÇELER"])
    aylik["İLÇELER"] = aylik["İLÇELER"].str.strip().replace(AD_DUZELT)
    aylik = aylik[aylik["İLÇELER"] != "GENEL TOPLAM"].set_index("İLÇELER")
    yillik[2023] = aylik[[str(a) for a in range(202301, 202313)]].sum(axis=1)
    return yillik / 1e6   # milyon m³


yillik = yillik_tuketim()
YILLAR = list(range(2011, 2023))

st.info(
    "ℹ️ İlçe tüketim verisi **yıllık** olarak yayımlanıyor (2023 hariç, o yıl aylık). Bu yüzden analizi yıllık yapıyoruz. "
    "Yıllık toplamı günlere bölüp günlük tahmin yapmak, her gün aynı değeri tekrar eden yapay bir seri üretir; "
    "böyle bir seride model gerçekte bir şey öğrenmeden mükemmel görünür."
)

# --- 1. Seçilen ilçenin yıllık tüketimi ---
st.header("1. İlçe bazında yıllık tüketim")
ilce = st.selectbox("İlçe seçin", sorted(yillik.index), index=sorted(yillik.index).index("ADALAR"))
seri = yillik.loc[ilce, YILLAR + [2023]].rename("Tüketim (milyon m³)")
seri.index = pd.to_datetime([f"{y}-01-01" for y in seri.index])
seri.index.name = "Yıl"
st.line_chart(seri, y_label="Milyon m³")
artis = (yillik.loc[ilce, 2022] / yillik.loc[ilce, 2011] - 1) * 100
c1, c2 = st.columns(2)
c1.metric("2022 tüketimi", f"{yillik.loc[ilce, 2022]:.2f} milyon m³")
c2.metric("2011 → 2022 değişim", f"%{artis:+.0f}")

# --- 2. En çok tüketen ilçeler ---
st.header("2. 2022'de en çok su tüketen ilçeler")
st.bar_chart(yillik[2022].sort_values(ascending=False).head(15), horizontal=True, x_label="Milyon m³")

# --- 3. 2023 tahmini: trend modeli taban modeli geçebiliyor mu? ---
st.header("3. 2023 tüketim tahmini: basit modeller")
st.markdown(
    "Her ilçe için 2011–2022 verisiyle 2023 tüketimini iki yolla tahmin edip gerçek 2023 değeriyle karşılaştırıyoruz:\n"
    "- **Taban (değişmez):** 2023 = 2022\n"
    "- **Doğrusal trend:** 2011–2022 yıllarına düz bir çizgi uydurup 2023'e uzatmak"
)
trend = yillik[YILLAR].apply(lambda s: np.polyval(np.polyfit(YILLAR, s.values, 1), 2023), axis=1)
gercek = yillik[2023]
mape = lambda t: (abs(t - gercek) / gercek).mean() * 100
sonuc = pd.DataFrame({"Ortalama yüzde hata (MAPE)": [mape(yillik[2022]), mape(trend)]},
                     index=["Taban: 2023 = 2022", "Doğrusal trend"])
st.dataframe(sonuc.style.format("%{:.1f}"), width="content")
st.caption(
    "Bulgu: yıllık ilçe tüketimi yıldan yıla çok az değişiyor; bu yüzden 'geçen yılın aynısı' tahmini trend modelinden "
    "daha isabetli. Karmaşık bir model kurmadan önce basit bir tabanla karşılaştırmanın neden önemli olduğunu gösteriyor."
)
detay = pd.DataFrame({"2022": yillik[2022], "Gerçek 2023": gercek, "Taban tahmini": yillik[2022], "Trend tahmini": trend})
st.dataframe(detay.loc[[ilce]].style.format("{:.2f}"), width="stretch")

st.caption("Veri: İBB Açık Veri Portalı, ilçe bazında su tüketim miktarları (m³). Grafiklerde milyon m³ kullanılmıştır.")
