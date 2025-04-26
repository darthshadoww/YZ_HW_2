import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import re
import unicodedata
import faiss
import os
import pickle
import torch
from scipy.stats import spearmanr

data = pd.read_excel("../../data/ogrenci_sorular_2025.xlsx")
LEN = 1000

sample_data = data.sample(n=LEN, random_state=42)

# def clean_text(text):
#     # HTML tag temizliği
#     text = re.sub(r'<[^>]+>', ' ', text)

#     # Sadece Türkçe harfler, rakamlar, noktalama işaretleri ve boşlukları koru
#     allowed_chars = r'[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ.,!? ]'
#     text = re.sub(allowed_chars, ' ', text)

#     # Fazla boşlukları azalt
#     text = re.sub(r'\s+', ' ', text)

#     # Unicode normalizasyonu
#     text = unicodedata.normalize('NFC', text)

#     return text.strip()

# def preprocess_matrix(data):
#   m,n = data.shape
#   new_data =[]

#   for i in range(m):
#     dummy = []
#     for j in range(n-1):
#       dummy.append(clean_data(data.iloc[i,j]))
#     dummy.append(data.iloc[i,j])
#     new_data.append(dummy)

#   return pd.DataFrame(new_data)


def split_data(data):
  gpt = np.concatenate((data.iloc[:,0], data.iloc[:,1]))
  deep = np.concatenate((data.iloc[:,0], data.iloc[:,2]))
  answers = np.array(data.iloc[:, 3])
  return gpt, deep, answers


def to_vectors(input, model_name='ytu-ce-cosmos/turkish-e5-large'):


  model = SentenceTransformer(model_name)
  embeddings = model.encode(input, convert_to_tensor=True, normalize_embeddings=True)


  scores = (embeddings[:LEN] @ embeddings[LEN:].T) * 100
  return embeddings, scores

def best_similarity(scores, k=5):
  m, n = scores.shape
  sim = []

  for i in range(m):
    row = scores[i]
    topk = torch.topk(row, k)
    topk_indices = topk.indices.cpu().numpy().tolist()
    sim.append(list(topk_indices))

  return np.array(sim)


def top1_5(sim, answers):
  m,n = sim.shape
  top1 = []
  top5 = []
  for i in range(m):
    top1.append(int(sim[i,0] == answers[i]))

    j = 0
    while(j < n and answers[i] != answers[sim[i,j]]):
      j+=1

    top5.append(int(j!=n))


  return np.array(top1), np.array(top5)


def correlation(top, answers):
  corr, p_value = spearmanr(top, answers)
  return corr, p_value







gpt, deep, answers = split_data(sample_data)
gpt_vector, gpt_scores = to_vectors(gpt)
deep_vector, deep_scores = to_vectors(deep)
gpt_sim = best_similarity(gpt_scores)
deep_sim = best_similarity(deep_scores)

gpt_top1, gpt_top5 = top1_5(gpt_sim, answers)
deep_top1, deep_top5 = top1_5(deep_sim, answers)

print(gpt_top1)