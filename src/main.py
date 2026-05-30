import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data_loader import load_config, load_and_preprocess_batadal
from automata import build_automata
from explainability import generate_explanation
from pipeline import train_deep_learning_model, create_sequences

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def evaluate_predictions(y_true, y_pred, model_name):
    """Metrikleri hesaplar ve ekrana basar."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"[{model_name}] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")
    return acc, prec, rec, f1

def plot_conf_matrix(y_true, y_pred, title, save_path):
    """Confusion Matrix çizer ve kaydeder."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_full_experiments():
    config = load_config()
    seeds = config['experiment']['random_seeds']
    
    base_dir = Path(__file__).resolve().parent.parent
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    batadal_data = load_and_preprocess_batadal()
    X_train_dl, y_train_dl, X_val_dl, y_val_dl, X_test_dl, y_test_dl = batadal_data["deep_learning"]
    X_train_pca, y_train_pca, X_val_pca, y_val_pca, X_test_pca, y_test_pca = batadal_data["automata"]

    time_steps = 10
    X_t_seq, y_t_seq = create_sequences(X_train_dl, y_train_dl, time_steps)
    X_v_seq, y_v_seq = create_sequences(X_val_dl, y_val_dl, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test_dl, y_test_dl, time_steps)
    input_size = X_t_seq.shape[2]

    # --- 1. DERİN ÖĞRENME DENEYLERİ ---
    print("\n--- DERİN ÖĞRENME (BLACK-BOX) TESTLERİ ---")
    for seed in seeds:
        print(f"\nSeed: {seed}")
        set_seed(seed)
        
        # LSTM
        lstm_model = train_deep_learning_model("LSTM", X_t_seq, y_t_seq, X_v_seq, y_v_seq, input_size)
        lstm_model.eval()
        with torch.no_grad():
            preds_lstm = lstm_model(torch.FloatTensor(X_test_seq).to(next(lstm_model.parameters()).device))
            preds_lstm_cls = (preds_lstm.cpu().numpy() > 0.5).astype(int).flatten()
        evaluate_predictions(y_test_seq, preds_lstm_cls, f"LSTM (Seed {seed})")
        if seed == 42: plot_conf_matrix(y_test_seq, preds_lstm_cls, "LSTM Confusion Matrix", plots_dir / "lstm_cm.png")

        # GRU
        gru_model = train_deep_learning_model("GRU", X_t_seq, y_t_seq, X_v_seq, y_v_seq, input_size)
        gru_model.eval()
        with torch.no_grad():
            preds_gru = gru_model(torch.FloatTensor(X_test_seq).to(next(gru_model.parameters()).device))
            preds_gru_cls = (preds_gru.cpu().numpy() > 0.5).astype(int).flatten()
        evaluate_predictions(y_test_seq, preds_gru_cls, f"GRU (Seed {seed})")
        if seed == 42: plot_conf_matrix(y_test_seq, preds_gru_cls, "GRU Confusion Matrix", plots_dir / "gru_cm.png")

    # --- 2. OTOMATA PARAMETRE ANALİZİ ---
    print("\n--- OTOMATA (WHITE-BOX) PARAMETRE ANALİZİ ---")
    windows = [3, 4, 5, 6]
    alphabets = [3, 4, 5, 6]
    
    for w in windows:
        for a in alphabets:
            print(f"\nOtomata Eğitiliyor -> Window Size: {w}, Alphabet Size: {a}")
            matrix, _ = build_automata(X_train_pca, window_size=w, alphabet_size=a)
            print(f"Toplam State Sayısı: {len(matrix)}")

import sys
import logging

# Profesyonel loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
<<<<<<< HEAD
    try:
        logging.info("Proje baslatiliyor... Veri setleri yukleniyor ve deneyler kosuluyor.")
        run_full_experiments()
        logging.info("Tebrikler! Tum deneyler ve model egitimleri basariyla tamamlandi.")
        
    except FileNotFoundError as e:
        logging.error(f"Kritik Hata: Veri seti veya konfigurasyon dosyasi eksik! Lutfen 'data/' klasorunu kontrol edin. Detay: {e}")
        sys.exit(1) # Sistemi guvenli bir sekilde kapat
        
    except Exception as e:
        logging.error(f"Sistemde beklenmeyen kritik bir hata olustu. Program guvenli sekilde durduruluyor. Detay: {e}")
        sys.exit(1)
    print("\nTüm deneyler başarıyla tamamlandı. Grafikler 'plots' klasörüne kaydedildi.")
=======
    run_full_experiments()
>>>>>>> 0b35961 (lokal degisiklikler guvenceye alindi)
