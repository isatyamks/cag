# CAG vs RAG — Live Benchmarking with KV Cache

> **Cache-Augmented Generation** meets **Retrieval-Augmented Generation** — head to head, query by query, in your terminal.

---

## Demo

<video src="Demo.mp4" controls width="100%"></video>

---

## What this actually is

This project is a direct, honest comparison between two ways of giving an LLM access to a document at inference time:

- **CAG (Cache-Augmented Generation)** — you pre-process the entire document *once*, save the model's internal KV cache to disk, and then reuse it for every query. The model never has to re-read the document again.
- **RAG (Retrieval-Augmented Generation)** — you use BM25 to pull the most relevant chunks out of the document at query time, then shove those chunks into the prompt fresh for every single query.

Every time you type a question, the project runs both approaches in parallel, prints a side-by-side terminal comparison with real latency numbers, and updates a live matplotlib chart of cumulative GPU token usage over your session.

No magic, no fluff. You can literally see the token counts and millisecond timings for every step.

---

## Why it matters

The classic RAG setup re-encodes context on every query. That's fast to get started, but costs compute on every single request. CAG flips that tradeoff — pay once upfront to build the cache, then nearly nothing per query (you only encode the new question tokens, not the whole document).

Whether CAG wins or loses for you depends on:
- How long your document is
- How many queries you're running against it
- Whether your GPU memory can hold the cache

This project makes that tradeoff *visible* with actual numbers, not theory.

---

## Project Structure

```
cag/
├── main.py                          # Thin launcher — loads model, doc, cache, starts session
├── requirements.txt
├── .env.example                     # Copy to .env and set HF_TOKEN
│
├── data/
│   └── India.txt                    # Sample document (swap this for anything)
│
└── src/
    ├── core/
    │   └── config.py                # Model name, cache dir, special tokens — auto-loads .env
    │
    ├── utils/
    │   └── env.py                   # Stdlib .env loader (no extra dependencies)
    │
    ├── llm/
    │   ├── loader.py                # Loads tokenizer + model from HuggingFace
    │   ├── prompts.py               # Chat template builders for CAG and RAG
    │   ├── generation.py            # Token-by-token greedy decode with timing
    │   └── tokens.py                # Token counting + cache size estimation
    │
    ├── cache/
    │   ├── manager.py               # Build cache or load from disk if it exists
    │   ├── store.py                 # Runs one forward pass to populate DynamicCache
    │   └── cleaner.py               # Trims the KV cache back to origin length
    │
    ├── rag/
    │   └── retriever.py             # BM25 retriever (no external dependencies)
    │
    ├── evaluation/
    │   ├── cag_runner.py            # CAGRunner — owns the full CAG execution path
    │   ├── rag_runner.py            # RAGRunner — owns the full RAG execution path
    │   ├── benchmark.py             # BenchmarkRunner — thin coordinator for both runners
    │   └── session.py               # BenchmarkSession — query loop + session state + plotter
    │
    └── visualization/
        ├── terminal.py              # Colored, boxed terminal output with ANSI codes
        └── plotter.py               # Live matplotlib chart of cumulative tokens
```

---

## How it works, step by step

### 1. Environment + model loading

On startup, `src/utils/env.py` automatically reads your `.env` file and injects `HF_TOKEN` into the environment — no manual `export` needed. Then `src/core/config.py` picks it up.

`src/llm/loader.py` pulls the tokenizer and model from HuggingFace. It automatically picks CUDA if available, falls back to CPU. Loads in `float16` on GPU and `float32` on CPU.

The default model is **Qwen2.5-3B-Instruct** — small enough to run on a single consumer GPU (even a 6GB VRAM card can handle it), but capable enough to give useful answers.

### 2. Building the KV cache (CAG setup)

`src/cache/store.py` runs one forward pass over the full document with the system prompt baked in, letting the model build its internal KV attention cache. That cache (a `DynamicCache` from HuggingFace transformers) gets saved to `cag_cache/<topic>.cache`.

Next time you run the same document, `src/cache/manager.py` detects the file and loads it straight from disk — skipping the expensive forward pass entirely.

`src/cache/cleaner.py` ensures the cache is trimmed back to exactly the original length after each query, so appended question tokens don't bleed into the next query's KV state.

### 3. Per-query CAG inference — `CAGRunner`

`src/evaluation/cag_runner.py` owns this path entirely:
1. Restore the KV cache to its original length
2. Tokenize `question + <|im_end|>...<|im_start|>assistant`
3. Run only those new tokens through the model (document context is already cached)
4. Greedy decode until EOS
5. Build and return the metrics dict

### 4. Per-query RAG inference — `RAGRunner`

`src/evaluation/rag_runner.py` owns this path entirely:
1. BM25 scores all document chunks against the question, picks top 2
2. Inject retrieved chunks into a fresh prompt alongside the question
3. Tokenize and encode the full prompt from scratch
4. Same greedy decoding
5. Build and return the metrics dict

### 5. Coordination — `BenchmarkRunner`

`src/evaluation/benchmark.py` is a thin coordinator. It holds references to `CAGRunner` and `RAGRunner` and delegates. That's it — 16 lines.

### 6. Session loop — `BenchmarkSession`

`src/evaluation/session.py` owns the interactive query loop, session-wide token totals, query count, and the live chart. `main.py` just calls `session.start()` and gets out of the way.

### 7. Terminal output + live chart

`src/visualization/terminal.py` prints a side-by-side comparison after every query:
- Both answers
- Token breakdown (question, context, cached, format overhead)
- Per-step latency (restore, tokenize, forward pass, decode)
- Session-wide totals and percentage of GPU work saved

`src/visualization/plotter.py` keeps a live matplotlib window showing cumulative GPU tokens vs query number for both approaches.

---

## Setup

### Prerequisites

- Python 3.10+
- A CUDA-capable GPU is strongly recommended (the model is 3B parameters)
- A HuggingFace account with access to the model

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set your HuggingFace token

Copy `.env.example` to `.env` and fill it in:

```bash
cp .env.example .env
```

Edit `.env`:

```
HF_TOKEN=hf_your_actual_token_here
```

That's it — the app auto-loads it on startup. No manual `export` needed.

### Run it

```bash
python main.py
```

The first run downloads the model and builds the KV cache (a few minutes). Every run after that loads the cache from disk and starts the benchmark loop immediately.

---

## Swapping the document

The default document is `data/India.txt`. To use your own:

1. Drop a `.txt` file into `data/`
2. In `main.py`, change the `topic` variable to match your filename (without the extension):

```python
topic = "your_document_name"
```

The cache filename is derived from `topic` automatically.

---

## Changing the model

In `src/core/config.py`, change `MODEL_NAME` to any HuggingFace causal LM that supports chat templates:

```python
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # bigger model, more VRAM
```

If you switch models, delete the old `.cache` file so a fresh one gets built for the new architecture.

---

## What the terminal output looks like

Every query produces something like this (with full color in your terminal):

```
┌──────────────────── Query #1  ·  CAG vs RAG ────────────────────┐
│  CAG (Cache-Augmented)        │  RAG (Retrieval-Augmented)       │
├───────────────────────────────┼──────────────────────────────────┤
│  Answer: India became         │  Answer: India became            │
│  independent in 1947          │  independent in 1947             │
├───────────────────────────────┼──────────────────────────────────┤
│  GPU Encoded    38  (this q)  │  GPU Encoded   312  (every q)    │
│    ↳ Question: 12             │    ↳ Question: 12                │
│    ↳ Format:   26             │    ↳ Context:  220               │
│                               │    ↳ Sys Text: 62                │
│  Cached       4821 [zero cost]│    ↳ Format:   18                │
│  Logical Ctx  4859            │                                  │
│  Output        12             │  Output        12                │
├───────────────────────────────┼──────────────────────────────────┤
│  Restore (Trim)      2 ms     │  Retrieval (BM25)    1 ms        │
│  Tokenize            1 ms     │  Tokenize            4 ms        │
│  Forward Pass      180 ms     │  Forward Pass      920 ms        │
│  Decode             85 ms     │  Decode             90 ms        │
│  Total Latency     268 ms     │  Total Latency    1015 ms        │
├──────────────────────────────────────────────────────────────────┤
│                    PERFORMANCE SUMMARY                           │
│                                                                  │
│  Encoding Saved (This Query): 274 tokens (87.8% less work)      │
│  Session CAG GPU Tokens:      38 tokens (new only)              │
│  Session RAG GPU Tokens:      312 tokens (re-encoded entirely)  │
│  Session Total Reduction:     87.8% (274 tokens never re-encoded)│
└──────────────────────────────────────────────────────────────────┘
```

---

## Known limitations

- **CPU is very slow** — the greedy decode loop is one token at a time, intentionally simple. Use CUDA.
- **Context length** — Qwen2.5-3B supports 32k tokens, which covers most long documents, but very large docs will hit the limit at cache-build time.
- **KV cache disk size** — the `.cache` file can be large (~30MB for the India.txt sample). The `cag_cache/` directory is gitignored by default.
- **BM25 retriever is simple** — no dense embeddings, no vector store. Fine for benchmarking; replace `SimpleRetriever` for production.
- **Greedy decoding only** — no sampling, no temperature. Clean for fair benchmarking comparisons, not for production generation.

---

## Repo hygiene

- `.env` is gitignored — your `HF_TOKEN` never gets pushed
- `cag_cache/`, `data/`, and `report_modular/` are gitignored — large or locally generated
- `.env.example` shows the exact variable names needed

---

## License

MIT — do whatever you want with it.
