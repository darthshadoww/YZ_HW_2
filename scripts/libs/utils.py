import matplotlib.pyplot as plt
from collections import Counter
metric_names = [
    "gpt top1",
    "gpt top5",
    "gpt corr",
    "gpt p-value",
    "deep top1",
    "deep top5",
    "deep corr",
    "deep p-value",
    "time"
]

def plot_distribution(answers):

  counter = Counter(answers)
  keys = [int(key) for key in counter.keys()]
  values = list(counter.values())

  plt.figure(figsize=(8, 6))
  bars = plt.bar(keys, values, color='skyblue')

  plt.title('Hangisi İyi Frekans Grafiği', fontsize=15)
  plt.xlabel('Değerler', fontsize=12)
  plt.ylabel('Frekans', fontsize=12)

  for bar in bars:
      height = bar.get_height()
      plt.text(bar.get_x() + bar.get_width()/2., height,
              f'{int(height)}',
              ha='center', va='bottom')

  plt.xticks(keys)
  plt.grid(axis='y', linestyle='--', alpha=0.7)

  plt.savefig('plot.png', dpi=300, bbox_inches='tight')

def plot_model_comparison_metrics(e5_metrics, cosmosE5_metrics, jina_metrics, metric_names):
    """
    3 model için 9 metriği gruplandırılmış çubuk grafiklerle karşılaştırır

    Parametreler:
    e5_metrics: e5 modeline ait 9 metrik değeri
    cosmosE5_metrics: cosmosE5 modeline ait 9 metrik değeri
    jina_metrics: jina modeline ait 9 metrik değeri
    metric_names: Metrik isimlerinin listesi (9 elemanlı)
    """
    models = ("e5", "cosmosE5", "jina")
    num_metrics = len(metric_names)

    # 3x3 grid oluştur (9 metrik için)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Model Performance Comparison (e5 vs cosmosE5 vs jina)', fontsize=16, y=1.02)

    # Her bir metrik için grafik oluştur
    for i, (ax, metric_name) in enumerate(zip(axes.flat, metric_names)):
        # Modellerin o metriğe ait değerleri
        values = (
            e5_metrics[i],
            cosmosE5_metrics[i],
            jina_metrics[i]
        )

        # Çubuk grafik oluştur
        bars = ax.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

        # Çubuk üstüne değerleri yaz
        ax.bar_label(bars, padding=3, fmt='%.3f')

        # Grafik başlığı ve düzenleme
        ax.set_title(metric_name)
        ax.set_ylim(0, max(values) * 1.2)  # Otomatik yükseklik ayarı

        # Sadece sol ve alt kenardaki grafiklere etiket ekle
        if i in [6, 7, 8]:
            ax.set_xlabel('Models')
        if i in [0, 3, 6]:
            ax.set_ylabel('Score')

    plt.tight_layout()
    plt.show()

