"""Ortak bölme ve metrikler: XGBoost ve LSTM aynı şekilde ölçülsün diye."""
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

UFUK = 30  # kaç gün sonrası tahmin ediliyor


def bol(veri, ufuk=UFUK):
    """Eğitim 2011-2022 (hedefi de 2022'de kalan satırlar), test 2023.
    Model ayarı için: egitim (2011-2020) + dogrulama (2021-2022)."""
    egitim_tum = veri[:pd.Timestamp("2022-12-31") - pd.Timedelta(days=ufuk)]
    test = veri["2023-01-01":pd.Timestamp("2023-12-31") - pd.Timedelta(days=ufuk)]
    return {"egitim_tum": egitim_tum, "egitim": egitim_tum[:"2020-12-31"],
            "dogrulama": egitim_tum["2021-01-01":], "test": test}


def metrikler(gercek, tahmin):
    """Doluluk yüzde puanı üzerinden MAE, RMSE, R2."""
    return {"MAE": mean_absolute_error(gercek, tahmin),
            "RMSE": mean_squared_error(gercek, tahmin) ** 0.5,
            "R2": r2_score(gercek, tahmin)}