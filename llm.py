import numpy as np
import re

sentences = [

    # --- Basic Statements ---
    "Fast learners understand concepts quickly.",
    "Slow learners need more time to grasp ideas.",
    "Smart learners ask good questions.",
    "Learning speed is different for everyone.",
    "Some students learn faster with practice.",

    # --- Paraphrases ---
    "Fast learners pick up new topics easily.",
    "Quick learners understand lessons rapidly.",
    "Slow learners improve steadily over time.",
    "Students learn at their own pace.",
    "Intelligent students enjoy solving problems.",

    # --- Cause and Effect ---
   
    "Fast learners review topics daily.",
    "Smart learners focus on understanding, not memorizing.",
    "Slow learners benefit from step-by-step learning.",
    "Good learners take notes and revise.",
    "Consistent study makes learners faster.",

    # --- Question/Answer Style (Very good for LLMs) ---
    "Q: Who learns quickly? A: Fast learners learn quickly.",
    "Q: Do slow learners fail? A: No, they improve with time.",
    "Q: What makes a learner smart? A: Curiosity and practice.",
    "Q: Is speed equal to intelligence? A: Not always.",
    "Q: How can slow learners improve? A: With consistency.",

    # --- Short Dialogue Style ---
    "Student: I learn slowly. Teacher: That is okay, keep practicing.",
    "Student: Fast learners are lucky. Teacher: Effort matters more than luck.",
    "Student: Can I become smarter? Teacher: Yes, learning builds intelligence.",
    "Student: I struggle with math. Teacher: Practice will help you improve.",
    "Student: I want to learn faster. Teacher: Focus and repetition help.",

    # --- Slight Complexity ---
    "Fast learners adapt quickly because they connect ideas faster.",
    "Slow learners often build stronger foundations with deep understanding.",
    "Smart learners do not fear mistakes because mistakes teach lessons.",
    "Learning quickly is useful, but learning correctly is better.",
    "Even smart learners need time to master difficult topics.",

    # --- General Knowledge Style ---
    "Learning is the process of gaining knowledge and skills.",
    "Intelligence can grow through effort and experience.",
    "The brain strengthens when learners practice regularly.",
    "Focus improves understanding and memory.",
    "Time and consistency create strong learners.",
]



def clean_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", "", s)  # remove punctuation
    return s.strip()

sentences = [clean_text(s) for s in sentences]

# =====================================================
# 2. Build Vocabulary
# =====================================================

words = sorted(set(" ".join(sentences).split()))

vocab = {w: i for i, w in enumerate(words)}
inv_vocab = {i: w for w, i in vocab.items()}

V = len(vocab)
print("Vocab size:", V)

# Tokenize dataset
data = [[vocab[w] for w in s.split()] for s in sentences]

# =====================================================
# 3. Hyperparameters
# =====================================================

d = 32              # embedding size
T = 50              # max sequence length
lr = 0.05
epochs = 100000      # reduce from 300k (too slow)

# =====================================================
# 4. Trainable Parameters
# =====================================================

E = np.random.randn(V, d) * 0.1       # token embeddings
P = np.random.randn(T, d) * 0.1       # positional embeddings

W_Q = np.random.randn(d, d) * 0.1
W_K = np.random.randn(d, d) * 0.1
W_V = np.random.randn(d, d) * 0.1

W_out = np.random.randn(d, V) * 0.1

# =====================================================
# 5. Helper Functions
# =====================================================

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)

def cross_entropy(probs, target):
    return -np.log(probs[target] + 1e-9)

def causal_mask(L):
    """Lower triangular mask for causal attention"""
    return np.tril(np.ones((L, L)))

# =====================================================
# 6. Training Loop (Only Output Layer Training)
# =====================================================

for epoch in range(epochs):

    total_loss = 0

    for seq in data:

        if len(seq) < 2:
            continue

        # Input = all except last token
        x_ids = seq[:-1]
        y_id  = seq[-1]

        L = len(x_ids)
        if L > T:
            x_ids = x_ids[:T]
            L = T

        # -----------------------------
        # Forward Pass
        # -----------------------------

        # Embedding + Position
        X = E[x_ids] + P[:L]   # (L, d)

        # QKV
        Q = X @ W_Q
        K = X @ W_K
        Vv = X @ W_V

        # Attention scores
        scores = (Q @ K.T) / np.sqrt(d)

        # Apply causal mask
        mask = causal_mask(L)
        scores = scores * mask + (-1e9) * (1 - mask)

        # Attention weights
        A = softmax(scores, axis=-1)

        # Output of attention
        H = A @ Vv

        # Last token representation
        h_last = H[-1]

        # Output logits → vocab probs
        logits = h_last @ W_out
        probs = softmax(logits)

        # Loss
        loss = cross_entropy(probs, y_id)
        total_loss += loss

        # -----------------------------
        # Backprop (Only W_out update)
        # -----------------------------

        dlogits = probs.copy()
        dlogits[y_id] -= 1

        dW_out = np.outer(h_last, dlogits)

        W_out -= lr * dW_out
        print(f"Epoch {epoch} | Loss: {total_loss}")

# =====================================================
# 7. Text Generation
# =====================================================

def generate(prompt, steps=5):

    prompt = clean_text(prompt)
    tokens = [vocab[w] for w in prompt.split() if w in vocab]

    for _ in range(steps):

        L = len(tokens)
        if L > T:
            tokens = tokens[-T:]
            L = T

        X = E[tokens] + P[:L]

        Q = X @ W_Q
        K = X @ W_K
        Vv = X @ W_V

        scores = (Q @ K.T) / np.sqrt(d)

        mask = causal_mask(L)
        scores = scores * mask + (-1e9) * (1 - mask)

        A = softmax(scores, axis=-1)

        H = A @ Vv
        h_last = H[-1]

        logits = h_last @ W_out
        probs = softmax(logits)

        next_id = np.argmax(probs)
        tokens.append(next_id)

    return " ".join(inv_vocab[i] for i in tokens)

print("\n--- Generation Test ---")
print(generate("fast learners", steps=5))