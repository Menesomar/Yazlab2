import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Yazdığımız modülleri içe aktarıyoruz
from data_loader import load_config, load_and_preprocess_batadal
from automata import build_automata
from explainability import generate_explanation

def set_seed(seed):
    """Deneylerin tekrarlanabilirliği için seed ayarlar."""
    np.random.seed(seed)
    # Eğer PyTorch kullanılacaksa buraya torch.manual_seed(seed) eklenebilir

def add_gaussian_noise(data, mean=0.0, std=0.1):
    """Veriye Gaussian gürültüsü ekler."""
    noise = np.random.normal(mean, std, size=data.shape)
    return data + noise

def plot_transition_heatmap(probability_matrix, save_path):
    """Geçiş olasılıklarını Heatmap olarak çizer ve kaydeder."""
    states = list(probability_matrix.keys())
    if not states:
        return
        
    # Matrisi 2D Numpy dizisine çevirme
    matrix_size = len(states)
    grid = np.zeros((matrix_size, matrix_size))
    
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    
    for current_state, transitions in probability_matrix.items():
        for next_state, prob in transitions.items():
            if next_state in state_to_idx:
                grid[state_to_idx[current_state], state_to_idx[next_state]] = prob
                
    plt.figure(figsize=(10, 8))
    sns.heatmap(grid, xticklabels=states, yticklabels=states, annot=False, cmap='YlGnBu')
    plt.title('Automata Transition Probability Heatmap')
    plt.xlabel('Next State')
    plt.ylabel('Current State')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Heatmap kaydedildi: {save_path}")

def run_experiments():
    config = load_config()
    seeds = config['experiment']['random_seeds'] # [42, 123, 2026, 7, 999] 
    
    print("Veriler yükleniyor...")
    batadal_data = load_and_preprocess_batadal()
    X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test = batadal_data["automata"]
    
    # Grafikler için klasör oluşturma
    base_dir = Path(__file__).resolve().parent.parent
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    for seed in seeds:
        print(f"\n{'='*40}")
        print(f"DENEY BAŞLIYOR - RANDOM SEED: {seed}")
        print(f"{'='*40}")
        set_seed(seed)
        
        # 1. Orijinal Veri ile Otomata İnşası [cite: 88]
        print("\n[Senaryo 1: Orijinal Veri]")
        prob_matrix, known_patterns = build_automata(
            X_train_pca, 
            window_size=config['model_params']['automata']['window_size'], 
            alphabet_size=config['model_params']['automata']['alphabet_size']
        )
        
        # Sadece ilk seed için Heatmap çizdir (kalabalık olmaması için)
        if seed == 42:
            plot_transition_heatmap(prob_matrix, plots_dir / "transition_heatmap_seed42.png")
            
        # 2. Gürültülü Veri Senaryosu 
        print("\n[Senaryo 2: Gürültülü Veri]")
        X_train_noisy = add_gaussian_noise(X_train_pca)
        noisy_prob_matrix, _ = build_automata(
            X_train_noisy,
            window_size=config['model_params']['automata']['window_size'],
            alphabet_size=config['model_params']['automata']['alphabet_size']
        )
        
        # 3. Açıklanabilirlik ve Unseen Testi
        print("\n[Açıklanabilirlik Modülü Testi]")
        # Test setinden rastgele bir örüntü alarak simülasyon yapıyoruz
        sample_pattern = "abc" 
        explanation = generate_explanation(
            time_step=1,
            current_state=known_patterns[0] if known_patterns else "aaa",
            incoming_pattern=sample_pattern,
            probability_matrix=prob_matrix,
            known_patterns=known_patterns
        )
        print(json.dumps(explanation, indent=4))

if __name__ == "__main__":
    run_experiments()
    print("\nTüm deneyler başarıyla tamamlandı. Grafikler 'plots' klasörüne kaydedildi.")