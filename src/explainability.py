import json
import numpy as np
from typing import Dict, List, Tuple, Union

def levenshtein_distance(s1: str, s2: str) -> int:
    """İki sembolik dizi (pattern) arasındaki düzenleme mesafesini hesaplar."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def handle_unseen_pattern(incoming_pattern: str, known_patterns: List[str]) -> Tuple[str, int]:
    """Unseen durumunda Levenshtein ile en yakın pattern'ı bulur."""
    distances = [levenshtein_distance(incoming_pattern, kp) for kp in known_patterns]
    min_index = int(np.argmin(distances))
    return known_patterns[min_index], distances[min_index]

def generate_explanation(
    time_step: int, 
    current_state: str, 
    incoming_pattern: str, 
    probability_matrix: Dict[str, Dict[str, float]], 
    known_patterns: List[str], 
    threshold: float = 0.15
) -> Dict[str, Union[int, str, float]]:
    """
    Olasılıksal otomata modelinin kararlarını deterministik olarak açıklar.

    Args:
        time_step (int): Zaman serisindeki mevcut adım.
        current_state (str): Otomatanın içinde bulunduğu aktif durum.
        incoming_pattern (str): SAX/PAA dönüşümünden gelen yeni örüntü.
        probability_matrix (Dict[str, Dict[str, float]]): Durumlar arası geçiş frekansları.
        known_patterns (List[str]): Eğitim setinde görülen benzersiz örüntüler sözlüğü.
        threshold (float): Anomali kararı için sınır olasılık değeri (varsayılan: 0.15).

    Returns:
        Dict[str, Union[int, str, float]]: Karar mekanizmasını özetleyen JSON uyumlu sözlük.
    """
    status = "seen"
    mapped_to = incoming_pattern
    
    # Unseen kontrolü
    if incoming_pattern not in known_patterns:
        status = "unseen"
        mapped_to, _ = handle_unseen_pattern(incoming_pattern, known_patterns)
        
    # Geçiş olasılığını matristen çek, yoksa (veya unseen ise) çok düşük bir olasılık ata
    prob = probability_matrix.get(current_state, {}).get(mapped_to, 0.001)
    
    # Düşük olasılıklı yollar anomali olarak işaretlenir
    decision = "anomaly" if prob < threshold else "normal"
    
    explanation = {
        "time_step": time_step,
        "state": current_state,
        "pattern": incoming_pattern,
        "status": status,
        "mapped_to": mapped_to if status == "unseen" else "-",
        "probability": round(prob, 3),
        "decision": decision
    }
    return explanation

def run_unit_tests() -> None:
    """Yönergede zorunlu tutulan Unseen yönetimi birim testleri."""
    print("Birim Testler (Unit Tests) Çalıştırılıyor...")
    
    # 1. Levenshtein Algoritması Testi
    assert levenshtein_distance("abc", "adc") == 1, "Levenshtein mesafe hesabı hatalı!"
    assert levenshtein_distance("aab", "abc") == 2, "Levenshtein mesafe hesabı hatalı!"
    
    # 2. Unseen Eşleme Testi
    known = ["abc", "bcc", "aab"]
    closest, dist = handle_unseen_pattern("adc", known)
    assert closest == "abc", f"Unseen eşleme hatalı! Beklenen 'abc', Bulunan '{closest}'"
    
    print("Tüm birim testler başarıyla geçti! ✓")

if __name__ == "__main__":
    run_unit_tests()
    
    # Yönergedeki [Örnek Açıklama] senaryosunun simülasyonu
    dummy_matrix = {"aab": {"abc": 0.108}}
    dummy_known = ["abc", "bcc", "aab"]
    
    result = generate_explanation(
        time_step=5, 
        current_state="aab", 
        incoming_pattern="adc", 
        probability_matrix=dummy_matrix, 
        known_patterns=dummy_known
    )
    
    print("\nModel Çıktısı (Zorunlu JSON Formatı):")
    print(json.dumps(result, indent=4))