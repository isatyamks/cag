import math
import re
from collections import Counter

class SimpleRetriever:
    def __init__(self, doc_text: str, chunk_size_words: int = 100, overlap_words: int = 20):
        # 1. Chunk the document
        words = doc_text.split()
        self.chunks = []
        step = chunk_size_words - overlap_words
        if step <= 0: step = chunk_size_words
        
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size_words])
            self.chunks.append(chunk)
            if i + chunk_size_words >= len(words):
                break
                
        # 2. Build term frequencies for BM25 algorithm
        self.chunk_words = [self._tokenize(c) for c in self.chunks]
        self.doc_freqs = Counter()
        for cw in self.chunk_words:
            for w in set(cw):
                self.doc_freqs[w] += 1
                
        self.N = len(self.chunks)
        self.avgdl = sum(len(cw) for cw in self.chunk_words) / self.N if self.N > 0 else 1

    def _tokenize(self, text: str) -> list:
        # Simple lowercase word tokenization
        return re.findall(r'\w+', text.lower())

    def retrieve(self, query: str, top_k: int = 2) -> str:
        q_words = self._tokenize(query)
        if not q_words or not self.chunks:
            return self.chunks[0] if self.chunks else ""
            
        scores = []
        k1 = 1.5
        b = 0.75
        
        for idx, cw in enumerate(self.chunk_words):
            score = 0.0
            cw_counts = Counter(cw)
            dl = len(cw)
            for qw in q_words:
                if qw not in cw_counts:
                    continue
                # Calculate IDF (Inverse Document Frequency)
                df = self.doc_freqs.get(qw, 0)
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                # Calculate TF (Term Frequency) with BM25 penalty for long chunks
                tf = cw_counts[qw]
                tf_score = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avgdl))
                score += idf * tf_score
            scores.append((score, self.chunks[idx]))
            
        # Sort chunks by relevance score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        best_chunks = [s[1] for s in scores[:top_k]]
        
        # Combine the best chunks with an ellipsis divider
        return "\n...\n".join(best_chunks)
