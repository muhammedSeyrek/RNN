"""
Vanilla RNN (Basit RNN) - Sıfırdan Implementasyon

Bu dosya, en temel RNN yapısını NumPy kullanarak sıfırdan oluşturur.
Her satır detaylı açıklamalarla anlatılmıştır.

RNN Formülleri:
--------------
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)  # Hidden state güncelleme
y_t = W_hy * h_t + b_y                          # Output hesaplama

Burada:
- h_t: t anındaki hidden state (gizli durum)
- x_t: t anındaki input (girdi)
- y_t: t anındaki output (çıktı)
- W_hh: Hidden-to-hidden ağırlık matrisi
- W_xh: Input-to-hidden ağırlık matrisi
- W_hy: Hidden-to-output ağırlık matrisi
"""

import numpy as np


class VanillaRNN:
    """
    Basit bir RNN sınıfı.

    Bu RNN:
    - Sekans verilerini işleyebilir
    - Her adımda önceki bilgiyi hatırlar
    - Backpropagation Through Time (BPTT) ile öğrenir
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        RNN'i başlat ve ağırlıkları initialize et.

        Parametreler:
        ------------
        input_size : int
            Girdi vektörünün boyutu
        hidden_size : int
            Hidden state vektörünün boyutu (RNN'in "hafıza" kapasitesi)
        output_size : int
            Çıktı vektörünün boyutu
        learning_rate : float
            Öğrenme hızı
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Ağırlıkları rastgele küçük değerlerle başlat
        # Xavier initialization kullanıyoruz (daha iyi başlangıç değerleri için)

        # W_xh: Input'tan hidden'a (input_size x hidden_size)
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01

        # W_hh: Hidden'dan hidden'a (hidden_size x hidden_size)
        # Bu ağırlık matrisi RNN'in "hafıza" özelliğini sağlar
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01

        # W_hy: Hidden'dan output'a (hidden_size x output_size)
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01

        # Bias terimleri (başlangıçta sıfır)
        self.b_h = np.zeros((1, hidden_size))  # Hidden bias
        self.b_y = np.zeros((1, output_size))  # Output bias

        # Gradient'lar için hafıza (BPTT için gerekli)
        self.reset_gradients()

    def reset_gradients(self):
        """Gradient'ları sıfırla"""
        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.dW_hy = np.zeros_like(self.W_hy)
        self.db_h = np.zeros_like(self.b_h)
        self.db_y = np.zeros_like(self.b_y)

    def tanh(self, x):
        """
        Tanh aktivasyon fonksiyonu.
        Çıktı aralığı: [-1, 1]
        """
        return np.tanh(x)

    def tanh_derivative(self, x):
        """
        Tanh'ın türevi (backpropagation için gerekli)
        d(tanh(x))/dx = 1 - tanh²(x)
        """
        return 1 - np.tanh(x) ** 2

    def softmax(self, x):
        """
        Softmax fonksiyonu (çıktıyı olasılık dağılımına çevirir)
        Numerically stable version
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, inputs):
        """
        Forward pass: Input sekansını işle ve output üret

        Parametreler:
        ------------
        inputs : list of numpy arrays
            Her biri (1, input_size) boyutunda olan input vektörleri
            Örnek: [x_1, x_2, x_3, ..., x_T]

        Dönüş:
        ------
        outputs : list of numpy arrays
            Her time step için output
        hidden_states : list of numpy arrays
            Her time step için hidden state (BPTT için gerekli)
        """
        T = len(inputs)  # Sekans uzunluğu (time steps)

        # Hidden states ve outputs'u sakla (BPTT için gerekli)
        hidden_states = []
        outputs = []

        # İlk hidden state sıfır (hafıza boş başlıyor)
        h_prev = np.zeros((1, self.hidden_size))

        # Her time step için
        for t in range(T):
            x_t = inputs[t]  # t anındaki input

            # Hidden state güncelle
            # h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            h_t = self.tanh(
                np.dot(x_t, self.W_xh) +      # Input katkısı
                np.dot(h_prev, self.W_hh) +   # Önceki hidden state katkısı (HAFIZA!)
                self.b_h                       # Bias
            )

            # Output hesapla
            # y_t = W_hy * h_t + b_y
            y_t = np.dot(h_t, self.W_hy) + self.b_y

            # Sakla
            hidden_states.append(h_t)
            outputs.append(y_t)

            # Bir sonraki adım için hidden state'i güncelle
            h_prev = h_t

        return outputs, hidden_states

    def backward(self, inputs, hidden_states, outputs, targets):
        """
        Backward pass: Backpropagation Through Time (BPTT)

        BPTT, RNN'lerde backpropagation'ın özel versiyonudur.
        Gradient'lar zaman boyunca geriye doğru yayılır.

        Parametreler:
        ------------
        inputs : list
            Input sekansı
        hidden_states : list
            Forward pass'den gelen hidden state'ler
        outputs : list
            Forward pass'den gelen output'lar
        targets : list
            Hedef değerler (ground truth)
        """
        T = len(inputs)
        self.reset_gradients()

        # Her time step için gradient hesapla (sondan başa doğru)
        dh_next = np.zeros((1, self.hidden_size))  # Bir sonraki adımdan gelen gradient

        for t in reversed(range(T)):
            x_t = inputs[t]
            h_t = hidden_states[t]
            y_t = outputs[t]
            target_t = targets[t]

            # Output layer gradient'ı
            # Loss = MSE (Mean Squared Error) kullanıyoruz
            dy = y_t - target_t  # (1, output_size)

            # W_hy gradient'ı
            self.dW_hy += np.dot(h_t.T, dy)
            self.db_y += dy

            # Hidden layer gradient'ı
            dh = np.dot(dy, self.W_hy.T) + dh_next  # (1, hidden_size)

            # Tanh'ın türevi üzerinden gradient
            # h_raw = W_xh * x_t + W_hh * h_{t-1} + b_h
            h_prev = hidden_states[t-1] if t > 0 else np.zeros((1, self.hidden_size))
            h_raw = (np.dot(x_t, self.W_xh) +
                     np.dot(h_prev, self.W_hh) +
                     self.b_h)

            dh_raw = dh * self.tanh_derivative(h_raw)  # (1, hidden_size)

            # Ağırlık gradient'ları
            self.dW_xh += np.dot(x_t.T, dh_raw)
            self.dW_hh += np.dot(h_prev.T, dh_raw)
            self.db_h += dh_raw

            # Bir önceki time step için gradient
            dh_next = np.dot(dh_raw, self.W_hh.T)

    def update_weights(self):
        """
        Gradient descent ile ağırlıkları güncelle
        """
        # Gradient clipping (exploding gradient problemini önlemek için)
        clip_value = 5.0
        for dparam in [self.dW_xh, self.dW_hh, self.dW_hy, self.db_h, self.db_y]:
            np.clip(dparam, -clip_value, clip_value, out=dparam)

        # Ağırlıkları güncelle
        self.W_xh -= self.learning_rate * self.dW_xh
        self.W_hh -= self.learning_rate * self.dW_hh
        self.W_hy -= self.learning_rate * self.dW_hy
        self.b_h -= self.learning_rate * self.db_h
        self.b_y -= self.learning_rate * self.db_y

    def train_step(self, inputs, targets):
        """
        Tek bir training step:
        1. Forward pass
        2. Backward pass
        3. Weight update

        Parametreler:
        ------------
        inputs : list
            Input sekansı
        targets : list
            Hedef değerler

        Dönüş:
        ------
        loss : float
            MSE loss
        """
        # Forward pass
        outputs, hidden_states = self.forward(inputs)

        # Loss hesapla (Mean Squared Error)
        loss = 0
        for y_t, target_t in zip(outputs, targets):
            loss += np.mean((y_t - target_t) ** 2)
        loss /= len(outputs)

        # Backward pass
        self.backward(inputs, hidden_states, outputs, targets)

        # Weight update
        self.update_weights()

        return loss

    def predict(self, inputs):
        """
        Tahmin yap (inference mode)

        Parametreler:
        ------------
        inputs : list
            Input sekansı

        Dönüş:
        ------
        predictions : list
            Tahminler
        """
        outputs, _ = self.forward(inputs)
        return outputs


# Test fonksiyonu
if __name__ == "__main__":
    print("=" * 60)
    print("Vanilla RNN Test")
    print("=" * 60)

    # Basit bir test
    input_size = 3
    hidden_size = 5
    output_size = 2

    # RNN oluştur
    rnn = VanillaRNN(input_size, hidden_size, output_size, learning_rate=0.01)

    # Dummy data (3 time step, her biri 3 boyutlu)
    inputs = [
        np.array([[1.0, 0.5, 0.2]]),  # t=0
        np.array([[0.8, 0.3, 0.1]]),  # t=1
        np.array([[0.6, 0.4, 0.3]])   # t=2
    ]

    targets = [
        np.array([[0.0, 1.0]]),
        np.array([[1.0, 0.0]]),
        np.array([[0.5, 0.5]])
    ]

    print(f"\nInput size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Output size: {output_size}")
    print(f"Sequence length: {len(inputs)}")

    # Birkaç iteration train et
    print("\nTraining...")
    for epoch in range(100):
        loss = rnn.train_step(inputs, targets)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Tahmin yap
    print("\nPredictions after training:")
    predictions = rnn.predict(inputs)
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        print(f"  Time {i}: Prediction={pred[0]}, Target={target[0]}")

    print("\n✅ RNN başarıyla çalışıyor!")
