"""
Basit Sekans Tahmini Örneği

Bu örnek, RNN'in basit bir sayı dizisini nasıl öğrendiğini gösterir.

Problem: Bir sonraki sayıyı tahmin et
Pattern: [0, 1, 2, 3, 4] → 5 gibi basit bir artış paterni

RNN bu paterni öğrenecek ve doğru tahminler yapacak!
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from models.vanilla_rnn import VanillaRNN
import matplotlib.pyplot as plt


def create_simple_sequence_data(seq_length=5, num_sequences=100):
    """
    Basit sekans verileri oluştur.

    Pattern: Her sekans [start, start+1, start+2, ..., start+seq_length-1]
    Hedef: Bir sonraki sayıyı tahmin et

    Parametreler:
    ------------
    seq_length : int
        Sekans uzunluğu
    num_sequences : int
        Kaç tane farklı sekans oluşturulsun

    Dönüş:
    ------
    X : list of lists
        Input sekansları
    y : list of lists
        Target sekansları
    """
    X = []
    y = []

    for _ in range(num_sequences):
        # Rastgele bir başlangıç noktası seç
        start = np.random.randint(0, 50)

        # Sekans oluştur: [start, start+1, start+2, ...]
        sequence = list(range(start, start + seq_length))

        # Normalize et (0-1 arasına getir, öğrenmeyi kolaylaştırır)
        sequence_normalized = [x / 100.0 for x in sequence]

        # Input ve target oluştur
        # Input: ilk (seq_length-1) eleman
        # Target: son (seq_length-1) eleman (her birinin bir sonrakini tahmin et)

        inputs = []
        targets = []

        for i in range(len(sequence_normalized) - 1):
            # Input: t anındaki değer
            inputs.append(np.array([[sequence_normalized[i]]]))
            # Target: t+1 anındaki değer (bir sonraki)
            targets.append(np.array([[sequence_normalized[i + 1]]]))

        X.append(inputs)
        y.append(targets)

    return X, y


def train_rnn_on_sequences():
    """
    RNN'i sekans tahmini için train et
    """
    print("=" * 70)
    print("🧠 RNN ile Basit Sekans Tahmini")
    print("=" * 70)

    # Hiperparametreler
    input_size = 1      # Her time step'te tek bir sayı
    hidden_size = 10    # RNN'in hafıza kapasitesi
    output_size = 1     # Tek bir sayı tahmin et
    learning_rate = 0.01
    epochs = 500
    seq_length = 5

    print(f"\n📊 Hiperparametreler:")
    print(f"   - Input size: {input_size}")
    print(f"   - Hidden size: {hidden_size}")
    print(f"   - Output size: {output_size}")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Sequence length: {seq_length}")

    # RNN oluştur
    rnn = VanillaRNN(input_size, hidden_size, output_size, learning_rate)

    # Training data oluştur
    print(f"\n📚 Veri oluşturuluyor...")
    X_train, y_train = create_simple_sequence_data(seq_length, num_sequences=100)
    print(f"   ✅ {len(X_train)} adet training sekansı oluşturuldu")

    # Örnek bir sekans göster
    print(f"\n📝 Örnek bir training sekansı:")
    example_seq = X_train[0]
    example_target = y_train[0]
    print(f"   Input sekansı: {[x[0][0] for x in example_seq]}")
    print(f"   Target sekansı: {[y[0][0] for y in example_target]}")

    # Training
    print(f"\n🏋️ Training başlıyor...")
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0

        # Her sekans üzerinde train et
        for inputs, targets in zip(X_train, y_train):
            loss = rnn.train_step(inputs, targets)
            epoch_loss += loss

        # Ortalama loss
        avg_loss = epoch_loss / len(X_train)
        losses.append(avg_loss)

        # Her 100 epoch'ta bir progress göster
        if (epoch + 1) % 100 == 0:
            print(f"   Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

    print(f"   ✅ Training tamamlandı!")

    # Test
    print(f"\n🧪 Test ediliyor...")
    print(f"   Yeni bir sekans oluşturalım ve RNN'in tahminlerini görelim:")

    # Test sekansı: [10, 11, 12, 13, 14]
    test_start = 10
    test_sequence = list(range(test_start, test_start + seq_length))
    test_sequence_normalized = [x / 100.0 for x in test_sequence]

    test_inputs = []
    for i in range(len(test_sequence_normalized) - 1):
        test_inputs.append(np.array([[test_sequence_normalized[i]]]))

    # Tahmin yap
    predictions = rnn.predict(test_inputs)

    print(f"\n   📈 Sonuçlar:")
    print(f"   {'Input':>10} | {'RNN Tahmini':>15} | {'Gerçek Değer':>15} | {'Hata':>10}")
    print(f"   {'-'*60}")

    for i, (input_val, pred, actual) in enumerate(zip(
        test_sequence_normalized[:-1],
        predictions,
        test_sequence_normalized[1:]
    )):
        pred_denorm = pred[0][0] * 100  # Denormalize et
        actual_denorm = actual * 100
        error = abs(pred_denorm - actual_denorm)

        print(f"   {input_val*100:>10.2f} | {pred_denorm:>15.2f} | {actual_denorm:>15.2f} | {error:>10.4f}")

    # Loss grafiği çiz
    print(f"\n📊 Loss grafiği oluşturuluyor...")
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('RNN Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    print(f"   ✅ Grafik 'training_loss.png' olarak kaydedildi!")

    print(f"\n{'='*70}")
    print(f"🎉 Başarılı! RNN basit sekans patternini öğrendi!")
    print(f"{'='*70}")

    return rnn, losses


def interactive_test(rnn):
    """
    Kullanıcıdan sekans alıp RNN'e tahmin yaptır
    """
    print("\n" + "="*70)
    print("🎮 İnteraktif Test Modu")
    print("="*70)
    print("\nRNN'e bir sekans verin, bir sonraki sayıyı tahmin etsin!")
    print("Örnek: 5, 6, 7, 8")
    print("Çıkmak için 'q' yazın\n")

    while True:
        user_input = input("Sekans girin (virgülle ayrılmış): ").strip()

        if user_input.lower() == 'q':
            print("Çıkılıyor...")
            break

        try:
            # Kullanıcı girdisini parse et
            sequence = [float(x.strip()) for x in user_input.split(',')]

            if len(sequence) < 2:
                print("❌ En az 2 sayı girin!")
                continue

            # Normalize et
            sequence_normalized = [x / 100.0 for x in sequence]

            # Input oluştur
            inputs = [np.array([[val]]) for val in sequence_normalized[:-1]]

            # Tahmin yap
            predictions = rnn.predict(inputs)
            last_prediction = predictions[-1][0][0] * 100  # Denormalize

            print(f"✅ RNN'in tahmini: {last_prediction:.2f}")
            print(f"   (Gerçek sonraki: {sequence[-1]:.2f})")
            print(f"   Hata: {abs(last_prediction - sequence[-1]):.2f}\n")

        except Exception as e:
            print(f"❌ Hata: {e}\n")


if __name__ == "__main__":
    # RNN'i train et
    rnn, losses = train_rnn_on_sequences()

    # İnteraktif test (opsiyonel - kullanıcı isterse)
    # interactive_test(rnn)
