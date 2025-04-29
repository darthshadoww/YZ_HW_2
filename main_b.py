import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import gc
from tqdm import tqdm
import os
import warnings

# Uyarıları ve paralel işlemleri kapat (transformers kütüphanesi için önemli)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

class E5Vectorizer:
    """Metinleri E5 modeli ile vektör temsillerine dönüştüren sınıf"""

    def __init__(self, model_name='intfloat/multilingual-e5-large-instruct'):
        """Başlatıcı fonksiyon
        Args:
            model_name: HuggingFace model ismi (varsayılan: multilingual-e5-large)
        """
        self.device = self._get_device()  # GPU/CPU otomatik seçimi
        self.model_name = model_name
        self.tokenizer, self.model = self._load_model()  # Model ve tokenizer yükleme

    def _get_device(self):
        """GPU varsa kullan, yoksa veya hata verirse CPU'ya geç"""
        if torch.cuda.is_available():
            try:
                # Basit bir tensör işlemiyle GPU testi
                _ = torch.tensor([1.0]).cuda()
                torch.cuda.empty_cache()
                print("GPU kullanılıyor")
                return torch.device("cuda")
            except:
                print("GPU mevcut ama hata veriyor, CPU kullanılıyor")
                return torch.device("cpu")
        else:
            print("GPU bulunamadı, CPU kullanılıyor")
            return torch.device("cpu")

    def _load_model(self):
        """HuggingFace modelini ve tokenizer'ını güvenli şekilde yükleme"""
        try:
            # Tokenizer'ı yükle
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # GPU varsa float16, CPU'da float32 kullan
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

            # Modeli düşük CPU bellek kullanımıyla yükle
            model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            ).to(self.device)

            print(f"Model başarıyla yüklendi ({self.device.type.upper()})")
            return tokenizer, model

        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            raise

    def encode(self, texts, batch_size=8):
        """Metin listesini vektörlere dönüştürür
        Args:
            texts: Vektörleştirilecek metin listesi
            batch_size: İşlem boyutu (GPU bellek sınırlarına göre ayarlanabilir)
        Returns:
            numpy.ndarray: Vektörleştirilmiş metinler
        """
        self.model.eval()  # Modeli eval moduna al
        all_embeddings = []  # Tüm vektörleri biriktirecek liste

        with torch.no_grad():  # Gradyan hesaplamasını kapat
            for i in tqdm(range(0, len(texts), batch_size), desc="Vektörleştirme"):
                batch = texts[i:i + batch_size]

                try:
                    # Metinlere 'query:' öneki ekleyip tokenize et (E5 modeli için gerekli)
                    inputs = self.tokenizer(
                        ["query: " + str(text) for text in batch],
                        padding=True,  # Farklı uzunluktaki metinler için padding
                        truncation=True,  # Maksimum uzunluk aşarsa kes
                        max_length=512,  # Maksimum token uzunluğu
                        return_tensors="pt"  # PyTorch tensörü olarak döndür
                    ).to(self.device)

                    # Model çıktılarını al
                    outputs = self.model(**inputs)

                    # E5 için son token'ın embedding'ini al (CLS token yerine)
                    last_token_pos = inputs['attention_mask'].sum(dim=1) - 1
                    embeddings = outputs.last_hidden_state[
                        torch.arange(len(batch)),
                        last_token_pos
                    ]

                    # Embedding'leri normalize et (cosine benzerliği için önemli)
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    all_embeddings.append(embeddings.cpu().numpy())

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Bellek hatası durumunda batch boyutunu yarıya indir
                        print(f"Bellek hatası! Batch boyutu {batch_size} -> {max(1, batch_size//2)}")
                        return self.encode(texts, batch_size=max(1, batch_size//2))
                    raise

        return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.array([])

def load_and_preprocess_data(filepath):
    """Excel dosyasından veriyi yükler ve temel temizlik yapar
    Args:
        filepath: Veri dosyası yolu
    Returns:
        pandas.DataFrame: İşlenmiş veri çerçevesi
    """
    df = pd.read_excel(filepath)

    # İlk 3 sütun metin, 4. sütun etiket olarak varsayılıyor
    text_cols = df.columns[:3]
    df[text_cols] = df[text_cols].fillna("")  # NaN değerleri boş string ile değiştir
    df[df.columns[3]] = df[df.columns[3]].astype(int)  # Etiketleri integer'a çevir

    return df

def calculate_vector_combinations(s, g, d):
    """Farklı vektör kombinasyonlarını hesaplar
    Args:
        s: Soru vektörleri
        g: Görüş vektörleri
        d: Değerlendirme vektörleri
    Returns:
        dict: Kombinasyon isimleri ve vektörler
    """
    return {
        's': s,  # Sadece soru vektörleri
        'g': g,  # Sadece görüş vektörleri
        'd': d,  # Sadece değerlendirme vektörleri
        's-g': s - g,  # Soru ve görüş farkı
        's-d': s - d,  # Soru ve değerlendirme farkı
        'g-d': g - d,  # Görüş ve değerlendirme farkı
        '|s-g|': np.abs(s - g),  # Mutlak fark
        '|s-d|': np.abs(s - d),
        '|s-g|-|s-d|': np.abs(s - g) - np.abs(s - d),  # Farkların farkı
        's+g': s + g,  # Toplam vektörler
        's+d': s + d,
        'g+d': g + d,
        's+g+d': s + g + d,  # Tüm vektörlerin toplamı
        'all': np.hstack([s, g, d, s-g, s-d, g-d, np.abs(s-g), np.abs(s-g)-np.abs(s-d)])
        # Tüm kombinasyonların birleşimi
    }

def evaluate_combinations(X_train, X_test, y_train, y_test, combinations):
    """Farklı vektör kombinasyonlarını değerlendir
    Args:
        X_train: Eğitim indeksleri
        X_test: Test indeksleri
        y_train: Eğitim etiketleri
        y_test: Test etiketleri
        combinations: Vektör kombinasyonları sözlüğü
    Returns:
        pandas.DataFrame: Değerlendirme sonuçları
    """
    results = []
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
    }

    for name, features in combinations.items():
        print(f"\nDeğerlendirilen kombinasyon: {name}")

        # Veriyi eğitim ve test olarak ayır
        X_train_sub = features[:len(X_train)]
        X_test_sub = features[len(X_train):]

        # Standardizasyon uygula
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sub)
        X_test_scaled = scaler.transform(X_test_sub)

        for model_name, model in models.items():
            # Modeli eğit ve değerlendir
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Metrikleri hesapla
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            # Sonuçları kaydet
            results.append({
                'combination': name,
                'model': model_name,
                'accuracy': acc,
                'f1_macro': report['macro avg']['f1-score']
            })

            print(f"{model_name} - Doğruluk: {acc:.4f}, F1: {report['macro avg']['f1-score']:.4f}")

    return pd.DataFrame(results)

def main():
    """Ana işlem akışı"""
    try:
        # 1. Veriyi yükle ve temizle
        print("Veri yükleniyor...")
        df = load_and_preprocess_data("ogrenci_sorular_2025.xlsx")

        # 2. Vektörleştiriciyi başlat
        print("\nModel yükleniyor...")
        vectorizer = E5Vectorizer()

        # 3. Metinleri vektörlere dönüştür
        print("\nMetinler vektörleştiriliyor...")
        s_vectors = vectorizer.encode(df.iloc[:, 0].tolist())  # 1. sütun: sorular
        g_vectors = vectorizer.encode(df.iloc[:, 1].tolist())  # 2. sütun: görüşler
        d_vectors = vectorizer.encode(df.iloc[:, 2].tolist())  # 3. sütun: değerlendirmeler

        # 4. Vektör kombinasyonlarını hesapla
        print("\nKombinasyonlar hesaplanıyor...")
        combinations = calculate_vector_combinations(s_vectors, g_vectors, d_vectors)

        # 5. Veriyi eğitim ve test olarak ayır
        print("\nVeri bölünüyor...")
        y = df.iloc[:, 3].values  # 4. sütun: etiketler
        X_train_idx, X_test_idx, y_train, y_test = train_test_split(
            np.arange(len(y)), y, test_size=0.2, random_state=42, stratify=y
        )

        # 6. Kombinasyonları değerlendir
        print("\nModel eğitimi başlıyor...")
        results = []
        for name, features in combinations.items():
            X_train = features[X_train_idx]
            X_test = features[X_test_idx]

            # Standardizasyon
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # RandomForest modeli
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Sonuçları kaydet
            acc = accuracy_score(y_test, y_pred)
            results.append({
                'combination': name,
                'accuracy': acc,
                'num_features': X_train.shape[1]  # Özellik sayısı
            })
            print(f"{name:15} | Doğruluk: {acc:.4f} | Boyut: {X_train.shape[1]}")

        # 7. Sonuçları göster
        results_df = pd.DataFrame(results)
        print("\nSonuçlar:")
        print(results_df.sort_values('accuracy', ascending=False))

    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
    finally:
        # Bellek temizleme
        if 'vectorizer' in locals():
            del vectorizer
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()