# CoVER-RAG

**CoVER-RAG: Evidence Sufficiency-Guided Coverage and Claim Verification for Multi-hop Question Answering**.

CoVER-RAG coordinates an evidence gap planner, an evidence state builder, and an evidence sufficiency evaluator. The evaluator checks fact coverage and verifies candidate claims against retrieved evidence. Failed checks guide further retrieval, up to the configured iteration limit.

Use `coverRAG` as the Python module and Hydra method name.

## Repository layout

```text
config/                      Hydra settings and local LLM routes
src/rag/coverRAG/            Main method, prompts, and evidence containers
src/rag/                    Baseline implementations
src/startup/                Corpus construction and evaluation runners
src/dataset/                Dataset loading and answer normalization
src/corpus/                 Retrieval corpus construction
src/evaluator/              Answer and retrieval metrics
server/                     Embedding, retrieval, reranking, and LLM services
scripts/                    Data download and experiment launchers
docs/                       Data preparation and running instructions
```

## Installation

Use Python **3.10.16**, a CUDA-compatible PyTorch installation, and NVIDIA GPUs for FAISS GPU retrieval and vLLM inference. GPU memory requirements depend on model size and context length.

```bash
conda create -n coverRAG python=3.10.16
conda activate coverRAG
python -m pip install '.[retrieve,llm]'
cp .env.example .env
```

Alternatively, with `uv` installed, use `uv sync --extra retrieve --extra llm`, then activate `.venv`.

## Prepare data and indexes

QA datasets, model weights, and generated FAISS indexes are external assets. See [data preparation](docs/DATA.md) for offline paths and corpus details.

```bash
python scripts/download_data.py --hotpot-corpus
```

Start the embedding service in a separate terminal:

```bash
CUDA_VISIBLE_DEVICES=0 bash server/embedder/run.sh
```

Build all three corpus caches, embeddings, and indexes from the repository root:

```bash
python main.py task=corpus dataset=hotpotqa
python main.py task=corpus dataset=2wikimultihopqa
python main.py task=corpus dataset=musique
```

Then start retrieval and reranking in separate terminals:

```bash
CUDA_VISIBLE_DEVICES=0 bash server/retriever/run.sh
CUDA_VISIBLE_DEVICES=0 bash server/rerank/run.sh
```

The retriever loads all three datasets at startup, so all three `.pkl` and `.index` pairs must be present. Service defaults are embedding `8010`, retrieval `8011`, and reranking `8012`.

## Start the language model

The default backbone and evaluation model are `Qwen/Qwen3-32B`. The client reads model routes from `config/litellm.yaml`.

```bash
CUDA_VISIBLE_DEVICES=1,2 PORT=8016 \
model=/path/to/Qwen3-32B served_model_name=Qwen/Qwen3-32B \
MAX_MODEL_LEN=8192 bash server/vllm/run_qwen.sh
```

Replace the checkpoint path with your local model directory or a Hugging Face model ID. The client disables Qwen3 thinking through `chat_template_kwargs`. See [running experiments](docs/RUNNING.md) for other models and service options.

## Run experiments

Choose 1,000 entries from a dataset's `dev.jsonl` and save their `id` values to a text file, one unique ID per line. Selection is up to you; the file order is preserved.

```bash
bash scripts/run_coverRAG.sh hotpotqa /path/to/hotpotqa.txt
```

To run all three datasets, prepare `hotpotqa.txt`, `2wikimultihopqa.txt`, and `musique.txt` in one directory, each containing 1,000 IDs:

```bash
bash scripts/run_all_coverRAG.sh /path/to/sample_ids
```

To use the entire development set, pass `all` instead of a file or directory. Use `DRY_RUN=1` to check launch settings without running inference.

Outputs are written to `outputs/<model-tag>/coverRAG/rag/<dataset>/<run>/`, including `evaluate.csv`, per-question JSON results, evidence traces, and LLM request logs. The runner computes answer metrics automatically.

## Offline checks

With the runtime dependencies installed, run:

```bash
python -m unittest discover -s tests -v
python main.py --cfg job
```
