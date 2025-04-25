# YZ_HW_2



```markdown
# BLM 3510 – Yapay Zeka 2024/2 – Ödev 2

## Proje Hakkında
Bu repo, 2024/2 döneminde BLM 3510 dersi için hazırlanan **Ödev 2**’nin tüm kod, veri ve raporlarını içerir.  
Ödev iki ana bölümden oluşuyor:
1. **Deney A – Sorudan Cevaba Başarı Analizi**  
   - 1000 rastgele soru seçilerek GPT4o ve Deepseek cevap vektörleri üzerinden **Top-1** ve **Top-5** başarı oranları hesaplanacak.  
   - Başarılar ile “hangisi iyi” (1,2,3,4) sınıf etiketleri arasındaki korelasyon incelenecek.

2. **Deney B – “Hangisi İyi” Sınıflandırma Modeli**  
   - Girdi olarak `s, g, d, s-g, s-d, g-d, |s-g|, |s-g|-|s-d|` vektörlerini alıp 1–4 etiketlerini tahmin eden modeller eğitilecek.  
   - Verisetinin %80’i eğitim, %20’si test olarak kullanılacak.  
   - Farklı vektör kombinasyonlarının sınıflandırma performansları karşılaştırılacak.

## Gereksinimler
- Python ≥ 3.8  
- [PyTorch](https://pytorch.org/)  
- [Transformers](https://github.com/huggingface/transformers)  
- pandas, numpy, scikit-learn, matplotlib  
- Internet bağlantısı (HuggingFace’ten önceden eğitilmiş modelleri indirmek için)

## Kurulum
```bash
git clone https://github.com/<kullanici_adiniz>/blm3510-odev2.git
cd blm3510-odev2
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Veri Seti
- Ham veriler Google Sheets’de:  
  `https://docs.google.com/spreadsheets/d/1Woh-A5oTJ715ivgIsu6NCkdav_k46iqr`  
- `data/` dizininde yer alan `questions.csv` ve `labels.csv` hali mevcut.  
- Dilerseniz `utils/download_data.py` ile doğrudan CSV’ye çekebilirsiniz.

## Deney A – Sorudan Cevaba Başarı Analizi
1. `scripts/retrieval_evaluation.py`  
   - Girdi: `--data data/questions.csv`  
   - Modeller: GPT4o ve Deepseek vektörleri (`--model gpt4o`, `--model deepseek`)  
   - Çıktı: Top-1 ve Top-5 başarı oranları, korelasyon analiz raporu  
2. Çıktılar `results/retrieval/` altında tablo ve grafik olarak saklanır.

## Deney B – “Hangisi İyi” Sınıflandırma Modeli
1. Kullanılan embedding modelleri:
   - `intfloat/multilingual-e5-large-instruct`  
   - `ytu-ce-cosmos/turkish-e5-large`  
   - `jinaai/jina-embeddings-v3`  
2. `scripts/classification.py`
   - Argümanlar:
     ```
     --data data/labels.csv
     --embeddings e5 cosmosE5 jina
     --features s,g,d,s-g,s-d,g-d,abs_s-g,delta_diff
     --split 0.8
     ```
   - Modeller: Logistic Regression, Random Forest, SVM (varsayılan)  
   - Sonuçlar `results/classification/` altına kaydedilir ve otomatik grafikler oluşturulur.

## Çalıştırma
```bash
# Deney A
python scripts/retrieval_evaluation.py \
  --data data/questions.csv \
  --models gpt4o deepseek

# Deney B
python scripts/classification.py \
  --data data/labels.csv \
  --embeddings e5 cosmosE5 jina \
  --features s,g,d,s-g,s-d,g-d,abs_s-g,delta_diff \
  --split 0.8
```

## Sonuçlar ve Grafikler
Tüm tablolar ve grafikler `results/` klasöründe yer alır. Öne çıkan bulgular:
- GPT4o vs. Deepseek Top-1/Top-5 başarı karşılaştırmaları  
- Top-başarı ile sınıf etiketleri arasındaki korelasyon  
- Farklı embedding kombinasyonlarının sınıflandırmadaki başarı farkları  

## Rapor ve Sunum
- **Rapor (PDF)**: `report/odev2_raporu.pdf`  
- **Sunum Videosu**: YouTube linkinizi `report/video_link.txt` dosyasına ekleyin.

## Proje Yapısı
```
├─ data/
│  ├─ questions.csv
│  └─ labels.csv
├─ scripts/
│  ├─ retrieval_evaluation.py
│  └─ classification.py
├─ results/
│  ├─ retrieval/
│  └─ classification/
├─ report/
│  ├─ odev2_raporu.pdf
│  └─ video_link.txt
├─ requirements.txt
└─ README.md
```

## Katkıda Bulunanlar
- **Melih Alçık** – 22011628  
- **Şahin Doğruca** – 22011049

## Lisans
Lisans yoktur
