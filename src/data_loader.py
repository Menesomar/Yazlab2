import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_config(config_path="config.yaml"):
    """Merkezi konfigürasyon dosyasını okur."""
    # Dosya yolu güvenliği için projenin ana dizini bulunuyor
    base_dir = Path(__file__).resolve().parent.parent
    full_config_path = base_dir / config_path
    with open(full_config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def get_data_paths():
    """İşletim sisteminden bağımsız (Mac/Windows uyumlu) klasör yollarını döndürür."""
    config = load_config()
    base_dir = Path(__file__).resolve().parent.parent
    skab_path = base_dir / config['data']['skab_path']
    batadal_path = base_dir / config['data']['batadal_path']
    return skab_path, batadal_path

# --- RIDVAN'IN SKAB KODLARI ---
def load_and_preprocess_skab():
    """SKAB veri setini okur, birleştirir ve özellikleri ayırır."""
    skab_path, _ = get_data_paths()
    data_frames = []
    
    for folder in ['valve1', 'valve2']:
        folder_path = skab_path / folder
        if not folder_path.exists():
            continue
            
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = folder_path / file_name
                df = pd.read_csv(file_path, sep=';', index_col=False) 
                
                df['source_group'] = folder
                df['source_file'] = file_name
                data_frames.append(df)
    
    if not data_frames:
        raise ValueError("SKAB verileri bulunamadı! Lütfen data/SKAB klasörünü kontrol edin.")
        
    full_df = pd.concat(data_frames, ignore_index=True)
    drop_cols = ['datetime', 'changepoint', 'source_group', 'source_file', 'anomaly']
    features = [col for col in full_df.columns if col not in drop_cols and col in full_df.columns]
    
    X = full_df[features]
    y = full_df['anomaly']
    groups = full_df['source_file']
    return X, y, groups

def get_skab_cv_splits(X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(X, y, groups))

# --- ENES'İN BATADAL KODLARI ---
def load_and_preprocess_batadal():
    """BATADAL verisini okur, zaman sıralı böler, ölçekler ve PCA uygular."""
    _, batadal_path = get_data_paths()
    
    # BATADAL klasöründeki csv dosyasını bul
    csv_files = [f for f in os.listdir(batadal_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("BATADAL klasöründe CSV dosyası bulunamadı! (Training Dataset 2 olmalı)")
        
    file_path = batadal_path / csv_files[0]
    df = pd.read_csv(file_path)
    
    # Zaman sütununu (DATETIME) tespit et ve modelden çıkar
    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    # Etiket sütununu tespit et (BATADAL'da genelde L_T1 veya ATT_FLAG olur)
    label_cols = [col for col in df.columns if 'label' in col.lower() or 'att' in col.lower() or 'l_t1' in col.lower()]
    target_col = label_cols[0] if label_cols else df.columns[-1]
    
    X = df.drop(columns=time_cols + [target_col])
    y = df[target_col]
    
    # %60 Eğitim, %20 Doğrulama, %20 Test (Zaman sırasını bozmadan)
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    # Sızıntı Önleme: Normalizasyon SADECE train'de fit edilir
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Otomata için PCA: SADECE train'de fit edilir, PC1'e indirgenir
    pca = PCA(n_components=1)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Derin Öğrenme (çok boyutlu) ve Otomata (tek boyutlu) için ayrı veri paketleri döndür
    return {
        "deep_learning": (X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test),
        "automata": (X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test)
    }

if __name__ == "__main__":
    print("Veri yükleyici modülü hazır. Fonksiyonlar başarıyla çağrılabilir.")