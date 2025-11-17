# 🧠 RNN Öğrenme Projesi

Recurrent Neural Networks (RNN) teknolojisini **sıfırdan** öğrenmek için hazırlanmış kapsamlı bir eğitim projesi.

## 📚 İçindekiler

1. [Proje Hakkında](#proje-hakkında)
2. [RNN Nedir?](#rnn-nedir)
3. [Kurulum](#kurulum)
4. [Adım Adım Öğrenme Yolu](#adım-adım-öğrenme-yolu)
5. [Proje Yapısı](#proje-yapısı)

---

## 🎯 Proje Hakkında

Bu proje, RNN'leri **teoriden pratiğe** öğrenmek için tasarlanmıştır. Her adımda:
- ✅ Teori açıklamaları
- ✅ Sıfırdan kod implementasyonu
- ✅ Pratik örnekler
- ✅ Görselleştirmeler

---

## 🤔 RNN Nedir?

**Recurrent Neural Network (RNN)**, **sekans verileriyle** çalışabilen özel bir sinir ağı türüdür.

### Normal NN vs RNN

```
Normal Neural Network (Feedforward):
Input → Hidden → Output
(Her girdi bağımsız)

Recurrent Neural Network:
Input₁ → Hidden₁ → Output₁
          ↓ (hafıza)
Input₂ → Hidden₂ → Output₂
          ↓ (hafıza)
Input₃ → Hidden₃ → Output₃
(Önceki adımları hatırlar!)
```

### RNN Kullanım Alanları

1. **Zaman Serisi Tahmini** 📈
   - Hisse senedi fiyat tahmini
   - Hava durumu tahmini
   - Sensör verileri analizi

2. **Doğal Dil İşleme (NLP)** 💬
   - Duygu analizi
   - Makine çevirisi
   - Metin üretimi

3. **Ses İşleme** 🎵
   - Konuşma tanıma
   - Müzik üretimi

4. **Video Analizi** 🎥
   - Hareket tanıma
   - Video sınıflandırma

---

## 🔧 Kurulum

```bash
# Gerekli paketleri yükle
pip install -r requirements.txt

# Jupyter Notebook başlat (opsiyonel)
jupyter notebook
```

---

## 📖 Adım Adım Öğrenme Yolu

### **Adım 1: Vanilla RNN** ⭐ (ŞU AN BURADAYIZ)
- 📁 `src/models/vanilla_rnn.py`
- 📁 `src/examples/simple_sequence.py`
- **Öğreneceklerimiz:**
  - RNN'in temel yapısı
  - Forward pass (ileri yayılım)
  - Backpropagation Through Time (BPTT)
  - Basit sekans tahmini

### **Adım 2: Vanishing Gradient Problemi**
- RNN'lerin en büyük sorunu
- Neden uzun sekansları öğrenemezler?
- Çözüm yolları

### **Adım 3: LSTM (Long Short-Term Memory)**
- Gate mekanizmaları
- Uzun vadeli bağımlılıkları öğrenme
- Pratik uygulamalar

### **Adım 4: GRU (Gated Recurrent Unit)**
- LSTM'in daha basit versiyonu
- Performans karşılaştırması

### **Adım 5: Gerçek Dünya Uygulamaları**
- Zaman serisi tahmini
- Metin sınıflandırma
- Sentiment analysis

---

## 📁 Proje Yapısı

```
RNN/
├── README.md                    # Bu dosya
├── requirements.txt             # Gerekli Python paketleri
│
├── src/
│   ├── models/                  # RNN model implementasyonları
│   │   ├── vanilla_rnn.py      # Basit RNN (Adım 1)
│   │   ├── lstm.py             # LSTM (Adım 3)
│   │   └── gru.py              # GRU (Adım 4)
│   │
│   ├── examples/                # Örnek uygulamalar
│   │   ├── simple_sequence.py  # Basit sekans tahmini
│   │   └── ...
│   │
│   └── utils/                   # Yardımcı fonksiyonlar
│       ├── data_loader.py
│       └── visualization.py
│
└── notebooks/                   # Jupyter notebook'lar
    ├── 01_vanilla_rnn.ipynb
    └── ...
```

---

## 🚀 Hızlı Başlangıç

```python
# Basit bir RNN örneği çalıştır
python src/examples/simple_sequence.py
```

---

## 📝 Notlar

- Her adım öncekinin üzerine inşa edilir
- Kodu çalıştırmadan önce teorik kısmı anlamaya çalışın
- Sorularınız için kod içindeki açıklamalara bakın

---

## 🎓 Kaynaklar

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Deep Learning Book - Sequence Modeling](https://www.deeplearningbook.org/)

---

**Başarılar! 🎉**
