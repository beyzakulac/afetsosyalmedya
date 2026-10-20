# İstanbul Baraj Doluluk Tahmini

Yapay Zeka dersi projesi: İstanbul barajlarının toplam doluluk oranını 30 gün sonrası için tahmin etmek.

- Veri: İBB Açık Veri Portalı (`data/raw/`)
- Bölme: eğitim 2011-2022, test 2023
- Metrikler: MAE, RMSE, R² (doluluk yüzde puanı)

## Kurulum
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Klasörler
- `data/raw/` ham Excel dosyaları (değiştirilmez)
- `data/processed/` temizlenmiş veri (git'e girmez, koddan üretilir)
- `notebooks/` keşif ve deneyler
- `src/` ortak kullanılan kod (temizleme, öznitelikler, değerlendirme)
- `app/` Streamlit paneli
