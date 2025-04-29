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

# Uyarıları ve paralel işlemleri kapat
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

class E5Vectorizer:
    def __init__(self, model_name='intfloat/multilingual-e5-large-instruct'):
        self.device = self._get_device()
        self.model_name = model_name
        self.tokenizer, self.model = self._load_model()

    def _get_device(self):
        """GPU varsa kullan, sorun çıkarsa CPU'ya geç"""
        if torch.cuda.is_available():
            try:
                # GPU testi
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
        """Modeli güvenli yükleme"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
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
        """Metinleri vektörlere dönüştürme"""
        self.model.eval()
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Vektörleştirme"):
                batch = texts[i:i + batch_size]

                try:
                    inputs = self.tokenizer(
                        ["query: " + str(text) for text in batch],
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(self.device)

                    outputs = self.model(**inputs)

                    # E5 için son token embedding'i
                    last_token_pos = inputs['attention_mask'].sum(dim=1) - 1
                    embeddings = outputs.last_hidden_state[
                        torch.arange(len(batch)),
                        last_token_pos
                    ]

                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    all_embeddings.append(embeddings.cpu().numpy())

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Bellek hatasında batch boyutunu yarıya indir ve yeniden dene
                        return self.encode(texts, batch_size=max(1, batch_size//2))
                    raise

        return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.array([])

def load_and_preprocess_data(filepath):
    """Veriyi yükle ve temizle"""
    df = pd.read_excel(filepath)

    # NaN değerleri boş string ile doldur
    text_cols = df.columns[:3]
    df[text_cols] = df[text_cols].fillna("")

    # Etiketleri sayısallaştır (1-4 arası)
    df[df.columns[3]] = df[df.columns[3]].astype(int)

    return df

def calculate_vector_combinations(s, g, d):
    """Vektör kombinasyonlarını hesapla"""
    return {
        's': s,
        'g': g,
        'd': d,
        's-g': s - g,
        's-d': s - d,
        'g-d': g - d,
        '|s-g|': np.abs(s - g),
        '|s-d|': np.abs(s - d),
        '|s-g|-|s-d|': np.abs(s - g) - np.abs(s - d),
        's+g': s + g,
        's+d': s + d,
        'g+d': g + d,
        's+g+d': s + g + d,
        'all': np.hstack([s, g, d, s-g, s-d, g-d, np.abs(s-g), np.abs(s-g)-np.abs(s-d)])
    }

def evaluate_combinations(X_train, X_test, y_train, y_test, combinations):
    """Kombinasyonları değerlendir"""
    results = []
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
    }

    for name, features in combinations.items():
        print(f"\nDeğerlendirilen kombinasyon: {name}")

        X_train_sub = features[:len(X_train)]
        X_test_sub = features[len(X_train):]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sub)
        X_test_scaled = scaler.transform(X_test_sub)

        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            results.append({
                'combination': name,
                'model': model_name,
                'accuracy': acc,
                'f1_macro': report['macro avg']['f1-score']
            })

            print(f"{model_name} - Doğruluk: {acc:.4f}, F1: {report['macro avg']['f1-score']:.4f}")

    return pd.DataFrame(results)

def main():
    try:
        # 1. Veriyi yükle
        print("Veri yükleniyor...")
        df = load_and_preprocess_data("ogrenci_sorular_2025.xlsx")

        # 2. Vektörleştiriciyi başlat
        print("\nModel yükleniyor...")
        vectorizer = E5Vectorizer()

        # 3. Metinleri vektörlere dönüştür
        print("\nMetinler vektörleştiriliyor...")
        s_vectors = vectorizer.encode(df.iloc[:, 0].tolist())
        g_vectors = vectorizer.encode(df.iloc[:, 1].tolist())
        d_vectors = vectorizer.encode(df.iloc[:, 2].tolist())

        # 4. Vektör kombinasyonlarını hesapla
        print("\nKombinasyonlar hesaplanıyor...")
        combinations = calculate_vector_combinations(s_vectors, g_vectors, d_vectors)

        # 5. Veriyi böl
        print("\nVeri bölünüyor...")
        y = df.iloc[:, 3].values
        X_train_idx, X_test_idx, y_train, y_test = train_test_split(
            np.arange(len(y)), y, test_size=0.2, random_state=42, stratify=y
        )

        # 6. Kombinasyonları değerlendir
        print("\nModel eğitimi başlıyor...")
        results = []
        for name, features in combinations.items():
            X_train = features[X_train_idx]
            X_test = features[X_test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            results.append({
                'combination': name,
                'accuracy': acc,
                'num_features': X_train.shape[1]
            })
            print(f"{name:15} | Doğruluk: {acc:.4f} | Boyut: {X_train.shape[1]}")

        # 7. Sonuçları göster
        results_df = pd.DataFrame(results)
        print("\nSonuçlar:")
        print(results_df.sort_values('accuracy', ascending=False))

    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
    finally:
        # Bellek temizle
        if 'vectorizer' in locals():
            del vectorizer
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()