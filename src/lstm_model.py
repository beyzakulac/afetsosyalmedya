import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. VERİYİ YÜKLEME
# Neden: Temizlenmiş verimizi modelin kullanabilmesi için Pandas ile okuyoruz.
df = pd.read_csv("data//temizlenmis_gunluk_tuketim.csv")

# Şimdilik sadece tek bir ilçe (ADALAR) üzerinden model mimarisini test ediyoruz.
df_ilce = df[df["İlçe"] == "ADALAR"].copy()

# 2. ÖLÇEKLEME (SCALING)
# Neden: Sinir ağları büyük sayılarla (örneğin 4000 m3) çalışırken zorlanır.
# Tüm su tüketim değerlerini 0 ile 1 arasına sıkıştırarak modelin çok daha hızlı ve stabil öğrenmesini sağlıyoruz.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_ilce[["Gunluk_Tuketim_m3"]])


# 3. PENCERELEME (WINDOWING)
# Neden: LSTM, geçmişe bakarak geleceği tahmin eder.
# Veriyi "geçmiş 30 günün değerleri (X)" ve "31. günün değeri (y)" şeklinde paketlere ayırıyoruz.
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : (i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


ZAMAN_ADIMI = 30  # Modele 30 günlük bir hafıza veriyoruz.
X, y = create_sequences(scaled_data, ZAMAN_ADIMI)

# Veriyi Eğitim (%80) ve Test (%20) olarak ikiye bölüyoruz
# Neden: Modelin daha önce hiç görmediği veriler üzerindeki performansını ölçmek için test verisi ayırmalıyız.
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 4. LSTM MODELİNİ İNŞA ETME
# Neden: Sıralı (Sequential) bir model kuruyoruz.
model = Sequential()

# İlk LSTM katmanı: 50 nöron içeriyor. return_sequences=False çünkü bir sonraki katman normal bir Dense katmanı.
# input_shape, modelin içine girecek verinin (30 gün, 1 özellik) boyutunu belirtir.
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))

# Dropout Katmanı (Opsiyonel ama kritik)
# Neden: Modelin veriyi ezberlemesini (overfitting) engellemek için eğitim sırasında rastgele %20 oranında nöronu kapatır.
model.add(Dropout(0.2))

# Çıktı Katmanı (Dense)
# Neden: Biz tek bir günün (31. gün) tüketimini tahmin etmek istediğimiz için çıktı katmanında 1 nöron (units=1) var.
model.add(Dense(units=1))

# 5. MODELİ DERLEME VE EĞİTME (COMPILE & FIT)
# Neden optimizer='adam': Yapay sinir ağlarında ağırlıkları güncelleyen en popüler ve verimli algoritmadır.
# Neden loss='mse' (Mean Squared Error): Regresyon (sayısal tahmin) problemlerinde hatayı ölçmek için en iyi yöntemdir.
model.compile(optimizer="adam", loss="mean_squared_error")

print("Model eğitimi başlıyor...")
# epochs=20: Model tüm eğitim verisini 20 kez baştan sona okuyup kendini geliştirecek.
# batch_size=32: Veriyi 32'şerli paketler halinde modele besliyoruz (RAM dostu).
history = model.fit(
    X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test)
)

print("Model eğitimi tamamlandı!")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 6. TAHMİN YAPMA VE TERS ÖLÇEKLEME (INVERSE TRANSFORM)
print("Test verisi üzerinden tahminler yapılıyor...")
y_pred_scaled = model.predict(X_test)

# Tahminleri ve gerçek değerleri 0-1 aralığından orijinal metreküp değerlerine geri döndürüyoruz
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_gercek = scaler.inverse_transform(y_test.reshape(-1, 1))

# 7. METRİKLERİ HESAPLAMA (Performans Ölçümü)
rmse = np.sqrt(mean_squared_error(y_test_gercek, y_pred))
mae = mean_absolute_error(y_test_gercek, y_pred)
print(f"Test RMSE (Hata Yön sapması): {rmse:.2f} m³")
print(f"Test MAE (Ortalama Mutlak Hata): {mae:.2f} m³")

# 8. MODELİ KAYDETME
# Arayüz (app) tarafında kullanmak üzere modeli kaydediyoruz
model.save("src/lstm_su_tahmin_modeli.keras")
print("Model 'src/lstm_su_tahmin_modeli.keras' olarak başarıyla kaydedildi!")

# 9. GÖRSELLEŞTİRME
plt.figure(figsize=(14, 5))
plt.plot(y_test_gercek, color="blue", label="Gerçek Su Tüketimi")
plt.plot(y_pred, color="red", label="LSTM Tahmini", alpha=0.7)
plt.title("ADALAR İlçesi - Gerçek vs Tahmin Edilen Günlük Su Tüketimi")
plt.xlabel("Zaman (Gün)")
plt.ylabel("Su Tüketimi (m³)")
plt.legend()
plt.grid(True)
plt.show()
