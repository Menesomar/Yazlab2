import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml
from pathlib import Path
import copy

def load_config(config_path="config.yaml"):
    """Merkezi konfigürasyon dosyasını okur."""
    base_dir = Path(__file__).resolve().parent.parent
    with open(base_dir / config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def create_sequences(X, y, time_steps=10):
    """Zaman serisi verisini derin öğrenme modelleri için 3 boyutlu dizilere çevirir."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y.iloc[i + time_steps] if hasattr(y, 'iloc') else y[i + time_steps])
    return np.array(Xs), np.array(ys)

# --- MODEL MİMARİLERİ (LSTM ve GRU) ---
class AnomalyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Sadece son zaman adımının çıktısını alıyoruz
        return torch.sigmoid(self.fc(out[:, -1, :]))

class AnomalyGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))

# --- EĞİTİM DÖNGÜSÜ ---
def train_deep_learning_model(model_name, X_train, y_train, X_val, y_val, input_size):
    """Konfigürasyondan okunan parametrelerle modeli eğitir ve Early Stopping uygular."""
    # Hard-coded engeli: Parametreler config'den çekiliyor
    config = load_config()['model_params']['deep_learning']
    epochs = config['max_epochs']
    batch_size = config['batch_size']
    patience = config['patience']

    # Cihaz seçimi (Windows için CUDA/CPU, Mac için MPS/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[{model_name}] Eğitim başlıyor... Kullanılan Cihaz: {device}")
    
    if model_name == "LSTM":
        model = AnomalyLSTM(input_size).to(device)
    elif model_name == "GRU":
        model = AnomalyGRU(input_size).to(device)
    else:
        raise ValueError("Desteklenmeyen model türü! Lütfen 'LSTM' veya 'GRU' seçin.")
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Verileri PyTorch Tensor'larına çevirme
    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).view(-1, 1).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.FloatTensor(y_val).view(-1, 1).to(device)

    train_data = TensorDataset(X_t, y_t)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False) # Zaman sırası bozulmaz

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation (Doğrulama) Aşaması
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_v)
            val_loss = criterion(val_outputs, y_v).item()
        
        # Early Stopping (Erken Durdurma) Mantığı
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{model_name}] Early stopping tetiklendi: Epoch {epoch+1}")
                break
                
    # En iyi ağırlıkları geri yükle
    model.load_state_dict(best_model_weights)
    print(f"[{model_name}] Eğitim tamamlandı. En iyi Val Loss: {best_val_loss:.4f}")
    return model

if __name__ == "__main__":
    print("Derin Öğrenme Pipeline'ı ve Modeller (LSTM, GRU) Hazır!")