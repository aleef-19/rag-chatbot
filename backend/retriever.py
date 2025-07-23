import os
import faiss
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRetriever:
    def __init__(self, doc_dir="backend/data/docs"):
        self.doc_dir = doc_dir
        self.docs = []
        self.doc_ids = []
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2")
        self.faiss_index = None
        self.embeddings= None
        self.load_docs() 
        self.build_indexes()

    def load_docs(self):
        for filename in os.listdir(self.doc_dir):
            with open(os.path.join(self.doc_dir, filename),"r",encoding="utf-8") as f:
                self.docs.append(f.read())
                self.doc_ids.append(filename)

    def build_indexes(self):
        #TF-IDF
        self.tfidf_matrix = self.tfidf.fit_transform(self.docs)

        #SBERT + FAISS
        self.embeddings = self.sbert.encode(self.docs, show_progress_bar=True)
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.faiss_index.add(self.embeddings)

    def retrieve(self, query, top_k = 3):
        #TF-IDF
        tfidf_vec = self.tfidf.transform([query])
        tfidf_scores = cosine_similarity(tfidf_vec,self.tfidf_matrix)[0]

        #SBERT
        query_emb = self.sbert.encode([query])
        D, I = self.faiss_index.search(query_emb, top_k)
        sbert_scores  = [(i, 1/(1 + d)) for i,d in zip(I[0],D[0])]

        #Combine (sum scores, normalize)
        hybrid_scores = {}
        for i , score in enumerate(tfidf_scores):
            hybrid_scores[i] = score

        for i, score in sbert_scores:
            hybrid_scores[i] = hybrid_scores.get(i, 0) + score

        ranked = sorted(hybrid_scores.items(),key=lambda x: x[1], reverse=True)[:top_k]
        return [self.docs[i] for i,_ in ranked]
    