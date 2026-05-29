# From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis

## Proje Tanımı ve Motivasyon
Bu çalışma, zaman serisi verileri üzerinde iki farklı modelleme paradigmasının (Derin Öğrenme tabanlı Black-Box modeller ile Sembolik Temsile dayalı White-Box Otomata modelleri) karşılaştırılmasını ve karar süreçlerinin olasılıksal olarak gerekçelendirilmesini amaçlamaktadır.

---

## 1. Veri Ön İşleme ve Veri Sızıntısı (Data Leakage) Yönetimi
Proje kapsamında anomali tespiti amacıyla SKAB ve BATADAL veri setleri kullanılmıştır. Veri sızıntısını önlemek için aşağıdaki katı kurallar uygulanmıştır:

* **SKAB Veri Seti:** Satır bazlı rastgele bölme zaman serisi bağımlılığını bozacağı için kullanılmamıştır. `source_file` sütunu grup değişkeni olarak alınarak **GroupKFold** stratejisi uygulanmıştır. Böylece aynı dosyaya ait kayıtların eğitim ve test kümesinde aynı anda yer alması engellenmiştir.
* **BATADAL Veri Seti:** Zaman sırası korunarak veri **%60 eğitim, %20 doğrulama ve %20 test** olarak ayrılmıştır.
* **Dönüşümler:** Veri normalizasyonu (StandardScaler) ve çok değişkenli veriyi tek boyuta indiren PCA (İlk Bileşen - PC1) dönüşümleri **yalnızca eğitim (train) verisi üzerinde fit edilmiş**, ardından doğrulama ve test verilerine uygulanmıştır.

---

## 2. Modelleme Yaklaşımları

### A. Derin Öğrenme Modelleri (Black-Box)
Sistemde PyTorch kullanılarak **LSTM** ve **GRU** mimarileri parametrik olarak kurgulanmıştır. Modeller merkezi konfigürasyondan beslenmekte olup şu sabit kurallarla eğitilmiştir:
* Epoch Üst Sınırı: 50
* Batch Size: 32
* Early Stopping: Validation loss takip edilerek (patience = 5)

### B. Olasılıksal Otomata Modeli (White-Box)
Otomata modeli şu dönüşüm adımları üzerinden inşa edilmiştir:
1. **PAA (Piecewise Aggregate Approximation)** ve **SAX (Symbolic Aggregate Approximation)** ile zaman serisi sembolik string dizilerine çevrilmiştir.
2. **Sliding Window** ile benzersiz örüntüler (patterns) çıkarılarak her biri bir "durum" (state) olarak tanımlanmıştır.
3. Durumlar arası geçiş olasılıkları frekans tabanlı olarak hesaplanmıştır:
   $$P(S_i \rightarrow S_j) = \frac{\text{Geçiş Sayısı}}{\text{Toplam Çıkış Sayısı}}$$

---

## 3. Unseen Pattern Yönetimi ve Birim Testler
Test aşamasında eğitim verisinde gözlemlenmemiş örüntülerle (unseen) karşılaşıldığında sistemin çökmesini önlemek amacıyla **Levenshtein (Edit Distance)** algoritması entegre edilmiştir. En yakın bilinen örüntü tespit edilerek sistemin bu durum üzerinden akışı sürdürmesi sağlanmıştır. Bu mekanizmanın doğruluğu `explainability.py` içerisindeki **birim testler (unit tests)** ile güvence altına alınmıştır.

---

## 4. Olasılıksal Açıklanabilirlik Modülü Çıktıları
Geliştirilen modül, modelin verdiği kararları deterministik ve yeniden üretilebilir şekilde gerekçelendirir. Her karar adımı için üretilen zorunlu **JSON formatı** aşağıdadır:

```json
{
    "time_step": 5,
    "state": "aab",
    "pattern": "adc",
    "status": "unseen",
    "mapped_to": "abc",
    "probability": 0.108,
    "decision": "anomaly"
}