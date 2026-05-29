import numpy as np
import scipy.stats as stats
from collections import defaultdict
import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
    base_dir = Path(__file__).resolve().parent.parent
    with open(base_dir / config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def get_sax_bins(alphabet_size):
    """SAX dönüşümü için normal dağılıma göre eşit olasılıklı sınırları (bins) belirler."""
    return stats.norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])

def ts_to_sax(window, window_size, bins):
    """Verilen zaman serisi penceresini PAA ve SAX dönüşümleri ile sembolik stringe çevirir."""
    # 1. PAA (Piecewise Aggregate Approximation)
    splits = np.array_split(window, window_size)
    paa = [np.mean(s) for s in splits]
    
    # 2. SAX (Symbolic Aggregate approXimation)
    indices = np.digitize(paa, bins)
    # 0, 1, 2 indekslerini 'a', 'b', 'c' harflerine dönüştür
    return "".join([chr(97 + i) for i in indices])

def build_automata(X_train, window_size=4, alphabet_size=3):
    """Eğitim verisi üzerinden Sliding Window ile state geçiş matrisini oluşturur."""
    print(f"Otomata İnşa Ediliyor (Window: {window_size}, Alphabet: {alphabet_size})...")
    bins = get_sax_bins(alphabet_size)
    
    patterns = []
    # Sliding Window: Her adımda 1 kaydırarak örüntü çıkarımı
    for i in range(len(X_train) - window_size):
        window = X_train[i : i + window_size]
        # PCA'dan gelen tek boyutlu veriyi düzleştir
        if isinstance(window, np.ndarray):
            window = window.flatten()
        pattern = ts_to_sax(window, window_size, bins)
        patterns.append(pattern)
        
    # State geçiş olasılıklarını hesaplama
    transitions = defaultdict(lambda: defaultdict(int))
    out_degrees = defaultdict(int)
    
    for i in range(len(patterns) - 1):
        current_state = patterns[i]
        next_state = patterns[i+1]
        
        transitions[current_state][next_state] += 1
        out_degrees[current_state] += 1
        
    probability_matrix = defaultdict(dict)
    
    # Formül: Geçiş Sayısı / Toplam Çıkış Sayısı
    for current_state, next_states in transitions.items():
        for next_state, count in next_states.items():
            probability_matrix[current_state][next_state] = count / out_degrees[current_state]
            
    return probability_matrix, patterns

if __name__ == "__main__":
    # Test amaçlı dummy veri
    dummy_data = np.random.randn(100)
    matrix, extracted_patterns = build_automata(dummy_data, window_size=4, alphabet_size=3)
    print(f"Toplam çıkarılan benzersiz state sayısı: {len(matrix)}")
    
    # Örnek bir state'in geçiş olasılıkları
    sample_state = list(matrix.keys())[0]
    print(f"Örnek State ({sample_state}) Geçiş Olasılıkları:")
    for next_st, prob in matrix[sample_state].items():
        print(f"  -> {next_st}: {prob:.3f}")