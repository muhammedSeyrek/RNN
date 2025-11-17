# 📉 Vanishing Gradient Problemi

## 🤔 Problem Nedir?

**Vanishing Gradient (Kaybolan Gradyan)**, RNN'lerin en büyük sorunudur. RNN'ler **uzun sekansları** öğrenemez çünkü gradient'lar zaman içinde geriye doğru yayılırken **çok küçülür** ve neredeyse sıfır olur.

---

## 🧮 Matematiksel Açıklama

RNN'de hidden state güncelleme formülü:

```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
```

### BPTT (Backpropagation Through Time)

Gradient'lar zaman içinde **geriye doğru** yayılır:

```
∂L/∂h_t = ∂L/∂h_{t+1} * ∂h_{t+1}/∂h_t

∂h_{t+1}/∂h_t = tanh'(·) * W_hh
```

**T time step geriye gidersek:**

```
∂L/∂h_0 = ∂L/∂h_T * (W_hh * tanh'(·))^T
```

### ⚠️ Problem:

1. **tanh'(x)** maksimum değeri **1**, genellikle **< 1**
2. **W_hh** genellikle küçük değerler içerir
3. Bu değerleri **T kez** çarpıyoruz

**Sonuç:** `(küçük_sayı)^T → 0` (T büyüdükçe)

---

## 🎯 Görsel Örnek

```
Time:     t=0  →  t=1  →  t=2  →  t=3  →  t=4

Gradient: ←---  ←---  ←---  ←---  ←---
          1.0   0.5   0.25  0.12  0.06

Gradient t=0'a ulaştığında neredeyse 0!
```

**t=0'daki** parametreler neredeyse **hiç güncellenmez** çünkü gradient çok küçük!

---

## 🔍 Neden Bu Kadar Önemli?

### Örnek: Dil Modeli

```
"Kedi bahçede oynarken, köpek evde uyurken, kuş ağaçta şarkı söylerken,
fare mutfakta koşarken, balık akvaryumda yüzerken, [BURAYA NE GELMELİ?]"
```

RNN bu uzun bağlamı **hatırlayamaz** çünkü başlangıçtaki bilgi **kaybolur**.

### Vanilla RNN'in Hafıza Kapasitesi

- ✅ **Kısa sekanslar** (5-10 step): İyi çalışır
- ⚠️ **Orta sekanslar** (10-20 step): Zorlanır
- ❌ **Uzun sekanslar** (20+ step): Öğrenemez

---

## 🧪 Pratik Deney

Aşağıdaki deneyde göreceğiz:

1. **Kısa sekans** (uzunluk=5): RNN öğrenir ✅
2. **Uzun sekans** (uzunluk=50): RNN öğrenemez ❌

---

## 💡 Çözümler

### 1. ❌ Geçici Çözümler (Yetersiz)
- **Gradient Clipping**: Gradient'ı sınırla (sadece exploding gradient'i çözer)
- **ReLU aktivasyonu**: Bazen yardımcı olur ama yetersiz
- **Xavier/He initialization**: Daha iyi başlangıç

### 2. ✅ Kalıcı Çözümler
- **LSTM (Long Short-Term Memory)**: Gate mekanizmaları ile gradient flow'u korur
- **GRU (Gated Recurrent Unit)**: LSTM'in daha basit versiyonu
- **Residual connections**: Skip connections ekler
- **Layer Normalization**: Gradient'ları stabilize eder

---

## 🔬 Exploding Gradient Problemi

**Ters durum:** Gradient'lar çok **büyür**

```
Gradient: ←---  ←---  ←---  ←---  ←---
          1.0   2.0   4.0   8.0   16.0  → ∞
```

**Çözüm:** Gradient Clipping (gradient'ı maksimum bir değere sınırla)

```python
# Bizim kodumuzdaki çözüm:
clip_value = 5.0
np.clip(gradient, -clip_value, clip_value)
```

---

## 📊 Özet

| Problem | Sebep | Sonuç | Çözüm |
|---------|-------|-------|-------|
| **Vanishing** | Gradient → 0 | Uzun bağımlılıklar öğrenilmez | LSTM/GRU |
| **Exploding** | Gradient → ∞ | Training instability | Gradient Clipping |

---

## 🚀 Sırada Ne Var?

Şimdi bu problemi **kod ile göreceğiz** ve **LSTM'in nasıl çözdüğünü** öğreneceğiz!

```
Adım 2: Vanishing Gradient (ŞU AN BURADAYIZ)
    ↓
Adım 3: LSTM (Çözüm!)
```
