"""Ham İBB verisini okuma ve temizleme."""
import pandas as pd

VERI = "data/raw/"

# İSKİ aktif kapasiteleri (milyon m3) - iski.istanbul sitesinden teyit edin
KAPASITE = {"Ömerli": 235.371, "Darlık": 107.500, "Elmalı": 9.600, "Terkos": 162.241, "Alibey": 34.143,
            "Büyükçekmece": 148.943, "Sazlıdere": 88.730, "Kazandere": 17.424, "Pabuçdere": 58.500, "Istrancalar": 6.231}


def doluluk_oku(klasor=VERI):
    """Baraj bazında günlük doluluk (0-1) + kapasiteye göre ağırlıklı 'toplam' (%)."""
    dol = pd.read_excel(klasor + "istanbul-dams-daily-occupancy-rates.xlsx")
    # 1.4.2023 - 8.8.2023 arası satırlar metin tarih (G.A.YYYY) ve yüzde olarak girilmiş
    metin = dol["Tarih"].map(lambda x: isinstance(x, str))
    dol.loc[metin, "Tarih"] = pd.to_datetime(dol.loc[metin, "Tarih"], format="%d.%m.%Y")
    dol["Tarih"] = pd.to_datetime(dol["Tarih"])
    barajlar = list(KAPASITE)
    dol.loc[metin, barajlar] = dol.loc[metin, barajlar] / 100
    dol = dol.set_index("Tarih").sort_index()
    k = pd.Series(KAPASITE)
    dol["toplam"] = (dol[k.index] * k).sum(axis=1) / k.sum() * 100
    return dol


def yagis_tuketim_oku(klasor=VERI):
    """Günlük ortalama yağış (mm, 10 havza ortalaması) ve İstanbul günlük tüketim (m3)."""
    yt = pd.read_excel(klasor + "istanbul-barajlarnda-ya-ve-gunluk-tuketim-verileri.xlsx")
    yt = yt.drop(columns=[c for c in yt.columns if str(c).startswith("Unnamed")]).set_index("Tarih").sort_index()
    yt = yt.rename(columns={"İstanbul günlük tüketim(m³/gün)": "tuketim"})
    yt["yagis"] = yt.drop(columns="tuketim").mean(axis=1)
    return yt[["yagis", "tuketim"]]


def birlesik_veri(klasor=VERI, bitis="2023-12-31"):
    """Doluluk + yağış + tüketim, günlük, 2011'den 'bitis' tarihine kadar."""
    df = doluluk_oku(klasor)[["toplam"]].join(yagis_tuketim_oku(klasor), how="inner")
    return df.asfreq("D").interpolate(limit=3)[:bitis]