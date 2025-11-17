"""
Vanishing Gradient Problemi - Pratik Deney

Bu kod, RNN'lerde vanishing gradient problemini gözle görülür şekilde gösterir.

Deney:
------
1. KISA sekans (5 step) ile train et → Başarılı ✅
2. UZUN sekans (50 step) ile train et → Başarısız ❌
3. Gradient büyüklüklerini karşılaştır
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from models.vanilla_rnn import VanillaRNN
import matplotlib.pyplot as plt


class RNNWithGradientTracking(VanillaRNN):
    """
    Gradient'ları takip eden RNN

    Normal RNN'e ek olarak, her time step'teki gradient büyüklüklerini kaydeder
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_history = []

    def backward(self, inputs, hidden_states, outputs, targets):
        """
        BPTT ile gradient hesapla ve gradient büyüklüklerini kaydet
        """
        T = len(inputs)
        self.reset_gradients()

        # Her time step için gradient büyüklüğünü kaydet
        gradient_norms = []

        dh_next = np.zeros((1, self.hidden_size))

        for t in reversed(range(T)):
            x_t = inputs[t]
            h_t = hidden_states[t]
            y_t = outputs[t]
            target_t = targets[t]

            # Output layer gradient
            dy = y_t - target_t

            # W_hy gradient
            self.dW_hy += np.dot(h_t.T, dy)
            self.db_y += dy

            # Hidden layer gradient
            dh = np.dot(dy, self.W_hy.T) + dh_next

            # Gradient büyüklüğünü kaydet (L2 norm)
            gradient_norm = np.linalg.norm(dh)
            gradient_norms.append(gradient_norm)

            # Tanh türevi
            h_prev = hidden_states[t-1] if t > 0 else np.zeros((1, self.hidden_size))
            h_raw = (np.dot(x_t, self.W_xh) +
                     np.dot(h_prev, self.W_hh) +
                     self.b_h)

            dh_raw = dh * self.tanh_derivative(h_raw)

            # Weight gradients
            self.dW_xh += np.dot(x_t.T, dh_raw)
            self.dW_hh += np.dot(h_prev.T, dh_raw)
            self.db_h += dh_raw

            # Bir önceki time step için gradient
            dh_next = np.dot(dh_raw, self.W_hh.T)

        # Gradient'ları ters çevir (t=0'dan t=T'ye sıralama)
        gradient_norms.reverse()
        self.gradient_history.append(gradient_norms)


def create_long_range_dependency_data(seq_length, num_sequences=50):
    """
    Uzun menzilli bağımlılık gerektiren sekanslar oluştur

    Pattern: İlk eleman → son eleman (aradaki elemanlar noise)

    Örnek:
    Input:  [1.0, 0.3, 0.7, 0.2, 0.5, ...]  (uzunluk=seq_length)
    Target: 1.0 (ilk elemanı hatırla!)

    Bu, RNN'in ilk elemanı "hatırlamasını" gerektirir.
    Sekans uzadıkça bu zorlaşır (vanishing gradient yüzünden).
    """
    X = []
    y = []

    for _ in range(num_sequences):
        # İlk eleman: hatırlanması gereken değer (0 veya 1)
        first_element = np.random.choice([0.0, 1.0])

        # Geri kalan elemanlar: random noise
        sequence = [first_element] + [np.random.uniform(0, 1) for _ in range(seq_length - 1)]

        # Input: tüm sekans
        inputs = [np.array([[val]]) for val in sequence]

        # Target: sadece son time step'te ilk elemanı tahmin et
        # (RNN'in ilk elemanı "hatırlaması" gerekir!)
        targets = [np.array([[0.0]])] * (seq_length - 1)  # Aradaki steplar için dummy target
        targets.append(np.array([[first_element]]))  # Son step: ilk elemanı tahmin et!

        X.append(inputs)
        y.append(targets)

    return X, y


def train_and_compare(short_seq_length=5, long_seq_length=50, epochs=200):
    """
    Kısa ve uzun sekansları karşılaştır
    """
    print("=" * 80)
    print("🧪 Vanishing Gradient Problemi - Pratik Deney")
    print("=" * 80)

    # Hiperparametreler
    input_size = 1
    hidden_size = 8
    output_size = 1
    learning_rate = 0.01

    results = {}

    # ========== KISA SEKANS ==========
    print(f"\n{'='*80}")
    print(f"📊 DENEY 1: KISA SEKANS (uzunluk={short_seq_length})")
    print(f"{'='*80}")

    # RNN oluştur
    rnn_short = RNNWithGradientTracking(input_size, hidden_size, output_size, learning_rate)

    # Veri oluştur
    X_short, y_short = create_long_range_dependency_data(short_seq_length, num_sequences=50)

    print(f"✅ {len(X_short)} adet sekans oluşturuldu")
    print(f"📝 Problem: İlk elemanı hatırla! (Sekans uzunluğu: {short_seq_length})")

    # Training
    print(f"\n🏋️ Training...")
    losses_short = []

    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, targets in zip(X_short, y_short):
            loss = rnn_short.train_step(inputs, targets)
            epoch_loss += loss

        avg_loss = epoch_loss / len(X_short)
        losses_short.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

    # Test
    print(f"\n🧪 Test ediliyor...")
    correct = 0
    for inputs, targets in zip(X_short[:10], y_short[:10]):
        predictions = rnn_short.predict(inputs)
        predicted_value = predictions[-1][0][0]  # Son step'in tahmini
        actual_value = targets[-1][0][0]

        # 0.5 threshold ile sınıflandır
        predicted_class = 1.0 if predicted_value > 0.5 else 0.0
        if predicted_class == actual_value:
            correct += 1

    accuracy_short = correct / 10.0
    print(f"✅ Doğruluk: {accuracy_short * 100:.0f}%")

    results['short'] = {
        'losses': losses_short,
        'accuracy': accuracy_short,
        'gradient_history': rnn_short.gradient_history[-10:]  # Son 10 batch
    }

    # ========== UZUN SEKANS ==========
    print(f"\n{'='*80}")
    print(f"📊 DENEY 2: UZUN SEKANS (uzunluk={long_seq_length})")
    print(f"{'='*80}")

    # RNN oluştur
    rnn_long = RNNWithGradientTracking(input_size, hidden_size, output_size, learning_rate)

    # Veri oluştur
    X_long, y_long = create_long_range_dependency_data(long_seq_length, num_sequences=50)

    print(f"✅ {len(X_long)} adet sekans oluşturuldu")
    print(f"📝 Problem: İlk elemanı hatırla! (Sekans uzunluğu: {long_seq_length})")

    # Training
    print(f"\n🏋️ Training...")
    losses_long = []

    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, targets in zip(X_long, y_long):
            loss = rnn_long.train_step(inputs, targets)
            epoch_loss += loss

        avg_loss = epoch_loss / len(X_long)
        losses_long.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

    # Test
    print(f"\n🧪 Test ediliyor...")
    correct = 0
    for inputs, targets in zip(X_long[:10], y_long[:10]):
        predictions = rnn_long.predict(inputs)
        predicted_value = predictions[-1][0][0]
        actual_value = targets[-1][0][0]

        predicted_class = 1.0 if predicted_value > 0.5 else 0.0
        if predicted_class == actual_value:
            correct += 1

    accuracy_long = correct / 10.0
    print(f"❌ Doğruluk: {accuracy_long * 100:.0f}%")

    results['long'] = {
        'losses': losses_long,
        'accuracy': accuracy_long,
        'gradient_history': rnn_long.gradient_history[-10:]
    }

    return results


def visualize_results(results, short_len, long_len):
    """
    Sonuçları görselleştir
    """
    print(f"\n{'='*80}")
    print("📊 Sonuçlar Görselleştiriliyor...")
    print(f"{'='*80}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss karşılaştırması
    ax = axes[0, 0]
    ax.plot(results['short']['losses'], label=f'Kısa Sekans (n={short_len})', linewidth=2)
    ax.plot(results['long']['losses'], label=f'Uzun Sekans (n={long_len})', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Karşılaştırması')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Accuracy karşılaştırması
    ax = axes[0, 1]
    accuracies = [results['short']['accuracy'] * 100, results['long']['accuracy'] * 100]
    bars = ax.bar(['Kısa Sekans', 'Uzun Sekans'], accuracies, color=['green', 'red'], alpha=0.7)
    ax.set_ylabel('Doğruluk (%)')
    ax.set_title('Test Doğruluğu')
    ax.set_ylim([0, 110])
    ax.axhline(y=50, color='gray', linestyle='--', label='Random Guess (50%)')
    ax.legend()

    # Bar'ların üzerine değerleri yaz
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 3. Gradient flow - Kısa sekans
    ax = axes[1, 0]
    # Son batch'in gradient'larını çiz
    if results['short']['gradient_history']:
        gradients = np.mean(results['short']['gradient_history'], axis=0)
        ax.plot(range(len(gradients)), gradients, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Time Step (geriden ileriye)')
        ax.set_ylabel('Gradient Büyüklüğü (L2 norm)')
        ax.set_title(f'Gradient Flow - Kısa Sekans (n={short_len})')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale (daha iyi görünür)

    # 4. Gradient flow - Uzun sekans
    ax = axes[1, 1]
    if results['long']['gradient_history']:
        gradients = np.mean(results['long']['gradient_history'], axis=0)
        ax.plot(range(len(gradients)), gradients, 'o-', linewidth=2, markersize=4, color='red')
        ax.set_xlabel('Time Step (geriden ileriye)')
        ax.set_ylabel('Gradient Büyüklüğü (L2 norm)')
        ax.set_title(f'Gradient Flow - Uzun Sekans (n={long_len}) ⚠️')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # İlk ve son gradient'ı vurgula
        first_grad = gradients[0]
        last_grad = gradients[-1]
        ax.axhline(y=first_grad, color='orange', linestyle='--', alpha=0.5, label=f'İlk: {first_grad:.2e}')
        ax.axhline(y=last_grad, color='purple', linestyle='--', alpha=0.5, label=f'Son: {last_grad:.2e}')
        ax.legend()

    plt.tight_layout()
    plt.savefig('vanishing_gradient_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Grafik 'vanishing_gradient_comparison.png' olarak kaydedildi!")

    # Özet
    print(f"\n{'='*80}")
    print("📋 ÖZET")
    print(f"{'='*80}")
    print(f"\n{'Metrik':<30} {'Kısa Sekans':>15} {'Uzun Sekans':>15}")
    print("-" * 80)
    print(f"{'Sekans Uzunluğu':<30} {short_len:>15} {long_len:>15}")
    print(f"{'Final Loss':<30} {results['short']['losses'][-1]:>15.6f} {results['long']['losses'][-1]:>15.6f}")
    print(f"{'Test Doğruluğu':<30} {results['short']['accuracy']*100:>14.1f}% {results['long']['accuracy']*100:>14.1f}%")

    if results['long']['gradient_history']:
        grad_ratio = results['long']['gradient_history'][-1][-1] / results['long']['gradient_history'][-1][0]
        print(f"{'Gradient Kaybı (son/ilk)':<30} {'-':>15} {grad_ratio:>15.6e}")

    print(f"\n{'='*80}")
    print("💡 SONUÇ")
    print(f"{'='*80}")
    print(f"✅ Kısa sekans: RNN başarılı (doğruluk={results['short']['accuracy']*100:.0f}%)")
    print(f"❌ Uzun sekans: RNN başarısız (doğruluk={results['long']['accuracy']*100:.0f}%)")
    print(f"\n🔍 Nedeni: Vanishing Gradient!")
    print(f"   Gradient'lar {long_len} step geriye giderken neredeyse sıfır oluyor.")
    print(f"   RNN ilk elemanı 'unutuyor' çünkü gradient oraya ulaşamıyor!")
    print(f"\n💡 Çözüm: LSTM veya GRU kullan (Adım 3'te göreceğiz!)")


if __name__ == "__main__":
    # Deneyi çalıştır
    short_length = 5
    long_length = 50

    results = train_and_compare(
        short_seq_length=short_length,
        long_seq_length=long_length,
        epochs=200
    )

    # Sonuçları görselleştir
    visualize_results(results, short_length, long_length)

    print(f"\n{'='*80}")
    print("🎉 Deney tamamlandı!")
    print(f"{'='*80}")
