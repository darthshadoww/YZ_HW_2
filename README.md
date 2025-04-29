# BLM 3510 Yapay Zeka – Ödev 2

**Sorudan Cevaba ve Cevap Kalitesi Sınıflandırma Analizi**

Bu repo, **Yıldız Teknik Üniversitesi - BLM 3510 Yapay Zeka dersi (2025/2 dönemi)** kapsamında hazırlanan **Ödev 2**'nin kaynak kodları, veri setleri, raporlar ve sonuçları içermektedir.

## 📌 Proje Özeti

Bu projede, yapay zekâ tarafından verilen cevapların kalitesini ve doğruluğunu analiz etmek üzere iki temel çalışma gerçekleştirilmiştir:

- **Deney A (Sorudan Cevaba Başarı Analizi):**

  - GPT-4o ve Deepseek modelleri tarafından verilen cevapların doğruluk oranları (Top-1 ve Top-5) analiz edilmiştir.
  - Başarı oranları ile kullanıcılar tarafından verilen kalite puanları arasındaki korelasyon incelenmiştir.

- **Deney B (Cevap Kalitesi Sınıflandırma):**
  - Soru ve cevap metinlerinden elde edilen vektörlerle cevap kalitesini sınıflandıran modeller eğitilmiştir.
  - Embedding modellerinin performansları karşılaştırmalı olarak incelenmiştir.

---

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler

- **Programlama Dili:** Python (≥ 3.8)
- **Makine Öğrenmesi Kütüphaneleri:**
  - PyTorch
  - Hugging Face Transformers
  - Scikit-learn
- **Veri Analizi ve Görselleştirme:**
  - pandas, numpy, matplotlib, seaborn
- **Embeddings:**
  - [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
  - [ytu-ce-cosmos/turkish-e5-large](https://huggingface.co/ytu-ce-cosmos/turkish-e5-large)
  - [jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3)

---

## 📂 Repo Yapısı

```
├── data/
│   ├── questions.csv        # Soru-cevap verileri
│   └── labels.csv           # Kalite sınıf etiketleri
│
├── scripts/
│   ├── retrieval_evaluation.py     # Deney A için analiz scripti
│   └── classification.py           # Deney B için model scripti
│
├── results/
│   ├── retrieval/                  # Deney A sonuçları (tablolar ve grafikler)
│   └── classification/             # Deney B sonuçları (tablolar ve grafikler)
│
├── report/
│   ├── odev2_raporu.pdf            # Detaylı analiz raporu
│   └── video_link.txt              # YouTube sunum linki
│
├── requirements.txt                # Bağımlılık dosyası
└── README.md                       # Bu belge
```

---

## 🚀 Projeyi Çalıştırma

### Adım 1 – Repoyu Klonlayın:

```bash
git clone https://github.com/<darthshadoww>/blm3510-yapayzeka-odev2.git
cd blm3510-yapayzeka-odev2
```

### Adım 2 – Ortamınızı Kurun:

```bash
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Adım 3 – Deneyleri Çalıştırın:

**🔸 Deney A – Başarı Analizi**

```bash
python scripts/retrieval_evaluation.py --data data/questions.csv --models gpt4o deepseek
```

**🔸 Deney B – Kalite Sınıflandırma**

```bash
python scripts/classification.py \
  --data data/labels.csv \
  --embeddings e5 cosmosE5 jina \
  --features s,g,d,s-g,s-d,g-d,abs_s-g,delta_diff \
  --split 0.8
```

---

## 📊 Veri Seti Kaynağı

Veri setinin orijinal kaynağına aşağıdaki linkten erişebilirsiniz:

[🔗 Veri Seti Google Sheets Bağlantısı](https://docs.google.com/spreadsheets/d/1Woh-A5oTJ715ivgIsu6NCkdav_k46iqr)

---

## 📑 Bulgular ve Sonuçlar

Tüm analiz sonuçları ve performans değerlendirmeleri detaylı olarak `results/` klasöründe sunulmuştur:

- GPT4o vs. Deepseek karşılaştırmalı Top-1 ve Top-5 doğrulukları
- Kalite sınıfı tahmin modellerinin performansları
- Embedding modellerinin performansa etkileri

Detaylı yorumlar ve analizler için `report/odev2_raporu.pdf` dosyasına bakınız.

---

## 🎥 Video Sunumu

Projenin anlatımını içeren sunum videosuna aşağıdaki bağlantıdan ulaşabilirsiniz:

[📺 Proje Sunumu (YouTube)](https://youtube.com/) _(Link buraya eklenecek.)_

---

## 👥 Katkıda Bulunanlar

| İsim          | Öğrenci No | E-posta                         |
| ------------- | ---------- | ------------------------------- |
| Melih Alçık   | 22011628   | melih.alcik@std.yildiz.edu.tr   |
| Şahin Doğruca | 22011049   | sahin.dogruca@std.yildiz.edu.tr |

---

## 📅 Teslim Tarihi ve Yöntemi

- **Son Teslim Tarihi:** 29 Nisan 2025, Saat 09:30
- **Teslim Yeri:** [online.yildiz.edu.tr](https://online.yildiz.edu.tr)

---

## 📜 Lisans Bilgisi

Bu proje, [MIT](LICENSE) lisansı altında sunulmuştur.

```text
MIT License © 2025
```
