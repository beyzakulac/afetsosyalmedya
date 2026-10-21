"""Ortak öznitelikler ve hazır modeller: notebook'lar ve panel aynı hesaplamayı kullanır."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.degerlendir import UFUK

# 02_sinem_xgboost.ipynb içindeki ızgara aramasında (doğrulama 2021-2022) en iyi çıkan ayar
XGB_AYAR = {"max_depth": 2, "learning_rate": 0.03, "n_estimators": 200, "min_child_weight": 1,
            "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 0}


def oznitelikler(df):
    """Sadece o güne kadar bilinen bilgiler: doluluk, son 7/30/90 gün değişim, yağış, tüketim, mevsim."""
    X = pd.DataFrame(index=df.index)
    X["doluluk"] = df["toplam"]
    for g in [7, 30, 90]:
        X[f"degisim_{g}"] = df["toplam"] - df["toplam"].shift(g)
        X[f"yagis_{g}"] = df["yagis"].rolling(g).sum()
        X[f"tuketim_{g}"] = df["tuketim"].rolling(g).mean() / 1e6
    X["mevsim_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    X["mevsim_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    return X


def model_verisi(df, ufuk=UFUK):
    """Öznitelikler + hedef (ufuk gün sonraki doluluk - bugünkü doluluk)."""
    X = oznitelikler(df)
    return X.assign(hedef=df["toplam"].shift(-ufuk) - df["toplam"]).dropna()


def modeller():
    """Karşılaştırmada kullanılan iki model (eğitilmemiş)."""
    return {"Doğrusal Regresyon": LinearRegression(), "XGBoost": XGBRegressor(**XGB_AYAR)}
