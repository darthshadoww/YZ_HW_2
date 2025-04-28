
import pandas as pd
import re
from transformers import AutoModel, AutoTokenizer
import torch
from annoy import AnnoyIndex
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import time
from libs import utils


emoji_pattern = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002700-\U000027BF"
    u"\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)
markdown_pattern = re.compile(r'(\*{1,3}|#{1,6}|---)')
multi_space_pattern = re.compile(r'\s+')


class CustomSentenceTransformer(torch.nn.Module):
    def __init__(self, model_name):
        super(CustomSentenceTransformer, self).__init__()

        # Use 'mps' (Metal) if available, else 'cpu'
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        # Load model with SentenceTransformer (required for E5)
        self.encoder = SentenceTransformer(model_name, device=self.device)

        # Load tokenizer separately (if needed for manual tokenization)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, sentences):
        # Let SentenceTransformer handle everything (tokenization + pooling)
        embeddings = self.encoder.encode(
            sentences,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=self.device
        )
        return embeddings.cpu()  # Return on CPU for compatibility

def clean_text(text):
    text = str(text)
    text = emoji_pattern.sub('', text)
    text = markdown_pattern.sub('', text)
    text = multi_space_pattern.sub(' ', text)
    return text.strip()

def clean_data(df):
    cleaned = df.iloc[:, :-1].apply(lambda x: x.map(clean_text))
    cleaned[df.columns[-1]] = df.iloc[:, -1]
    return cleaned

def encode_sentences(model, sentences, batch_size=64):
    all_embeddings = []

    model.eval()

    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            embeddings = model(batch)
            all_embeddings.append(embeddings)

    return torch.cat(all_embeddings, dim=0)





def search(query_embeddings, candidate_embeddings, k=5, n_trees=10):
    """
    Annoy kullanarak en benzer k adeti bulur.

    Args:
        query_embeddings (torch.Tensor): Sorgu embeddingleri [num_queries, hidden_size].
        candidate_embeddings (torch.Tensor): Aranacak cevap embeddingleri [num_candidates, hidden_size].
        k (int): Her sorgu için kaç adet en yakın sonucu döndürecek.
        n_trees (int): Annoy'da kaç ağaç kullanılacağı (default: 10)

    Returns:
        D (np.ndarray): Benzerlik skorları [num_queries, k].
        I (np.ndarray): Bulunan indexler [num_queries, k].
    """
    d = query_embeddings.shape[1]

    # 1. Annoy index oluştur (angular = cosine similarity gibi davranır)
    index = AnnoyIndex(d, 'angular')

    # 2. Candidate embedding'leri indexe ekle
    for i in range(candidate_embeddings.shape[0]):
        index.add_item(i, candidate_embeddings[i].cpu().numpy())

    # 3. Indexi inşa et
    index.build(n_trees)

    # 4. Sorgu başına arama yap
    all_indices = []
    all_distances = []

    for i in range(query_embeddings.shape[0]):
        idxs, dists = index.get_nns_by_vector(
            query_embeddings[i].cpu().numpy(),
            k,
            include_distances=True
        )
        all_indices.append(idxs)
        all_distances.append(dists)

    # 5. Numpy array'lere çevir
    I = np.array(all_indices)
    D = np.array(all_distances)

    return D, I



def top1_5(indexes, answers):
    top1 = []
    top5 = []
    for i in range(len(indexes)):
        if indexes[i][0] < len(answers):
            top1.append(int(answers[indexes[i][0]] == answers[i]))
        else:
            top1.append(0)

        found = False
        for idx in indexes[i]:
            if idx < len(answers) and answers[idx] == answers[i] and found == False:
                found = True
        top5.append(int(found))
    return top1, top5

def correlation(top1_list, top5_list):
    """
    Top-1 ve Top-5 sonuçlarının Spearman korelasyonunu hesaplar.

    Args:
        top1_list (list or np.ndarray): Her query için top-1 skorları.
        top5_list (list or np.ndarray): Her query için top-5 skorları.

    Returns:
        correlation (float): Spearman sıralama korelasyonu.
        p_value (float): İstatistiksel anlamlılık p-değeri.
    """
    # NumPy array'e çevir
    top1_array = np.array(top1_list)
    top5_array = np.array(top5_list)

    # Spearman korelasyonu hesapla
    correlation, p_value = spearmanr(top1_array, top5_array)

    return correlation, p_value

def run_expA(model_name):
  results = []
  data = pd.read_excel("data/ogrenci_sorular_2025.xlsx")
  sample_data = data.sample(n=1000, random_state=24)

  cleaned_data = clean_data(sample_data)

  model = CustomSentenceTransformer(model_name)

  question_embeddings = encode_sentences(model, cleaned_data.iloc[:,0].tolist(), batch_size=64)
  gpt4o_embeddings = encode_sentences(model, cleaned_data.iloc[:,1].tolist(), batch_size=64)
  deepseek_embeddings = encode_sentences(model, cleaned_data.iloc[:,2].tolist(), batch_size=64)

  distances, indexes = search(question_embeddings, gpt4o_embeddings)

  answers = np.array(cleaned_data.iloc[:,3])

  top1, top5 = top1_5(indexes, answers)
  print(f'gpt top1: {sum(top1)/len(top1)}')
  print(f"gpt top5: {sum(top5)/len(top5)}")
  results.append(top1)
  results.append(top5)

  correlation, p_value = correlation(top1, top5)

  results.append(correlation)
  results.append(p_value)
  print(f"gpt correlation: {correlation:.4f}")
  print(f"gpt p-value: {p_value:.4f}")


  distances, indexes = search(question_embeddings, deepseek_embeddings)

  top1, top5 = top1_5(indexes, answers)
  print(f'deep top1: {sum(top1)/len(top1)}')
  print(f"deep top5: {sum(top5)/len(top5)}")
  results.append(top1)
  results.append(top5)


  correlation, p_value = correlation(top1, top5)

  print(f"deep correlation: {correlation:.4f}")
  print(f"deep p-value: {p_value:.4f}")
  results.append(correlation)
  results.append(p_value)

  return results, answers

results = []

start = time.time()
result, answers = run_expA('intfloat/multilingual-e5-large-instruct')
end = time.time()
result.append(end-start)
results.append(result)


# start = time.time()
# result = run_expA('ytu-ce-cosmos/turkish-e5-large')
# end = time.time()
# result.append(end-start)
# results.append(result)

# start = time.time()
# result, answers = run_expA("jinaai/jina-embeddings-v3")
# end = time.time()
# result.append(end-start)
# results.append(result)

utils.plot_distribution(answers)