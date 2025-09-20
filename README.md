```markdown
# Conversation Embedding (GPU)

This project ingests conversation data (utterance chunks plus speaker analytics), generates vector embeddings using Sentence Transformers with optional GPU acceleration, and stores them in a Qdrant vector database. It can also group contiguous utterances by speaker (segments) and provides a simple similarity search API over the stored content.

## Features

- GPU-accelerated embedding generation with automatic CPU fallback
- Batch encoding for efficient throughput
- Rich payload metadata for each stored item (speaker info, timestamps, word counts, etc.)
- Speaker-segment creation and storage in addition to per-chunk storage
- Processed-file tracking to skip unchanged inputs
- Simple semantic search with optional metadata filtering
- Collection stats and processed-file status

## Project Structure

- src/__init__/Conversation Embedding Code.py — main script and library class
- src/__init__/__init__.py — package initializer
- requirements.txt — Python dependencies
- README.md — this file

## Prerequisites

- Python 3.10+ recommended
- Qdrant running locally (default URL: http://localhost:6333)
  - Docker: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
- Optional GPU acceleration
  - NVIDIA GPU with compatible CUDA drivers
  - CUDA-enabled PyTorch build

## Installation

Using a Conda environment is recommended for isolation and GPU compatibility.
```
bash
# Create and activate a conda environment
conda create -n conv-embed python=3.10 -y
conda activate conv-embed

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Dependencies include (see requirements.txt for exact versions):
- torch, torchvision, torchaudio
- sentence-transformers
- transformers, accelerate
- qdrant-client
- numpy

Tip: For CUDA-enabled PyTorch, follow the official PyTorch install matrix to match your CUDA/driver version.

## Data Layout

Prepare a base directory of conversation documents. Each document is a folder containing two JSON files:

- {document_name}_conversation_chunks.json
- {document_name}_speaker_analytics.json

Example:
- Conversation_Chunks/
  - ProjectKickoff/
    - ProjectKickoff_conversation_chunks.json
    - ProjectKickoff_speaker_analytics.json
  - PerformanceReview/
    - PerformanceReview_conversation_chunks.json
    - PerformanceReview_speaker_analytics.json

Expected content overview:
- conversation_chunks.json: conversations keyed by conversation_id with chunk payloads (text, speaker, timestamp, chunk_index) and document_info.
- speaker_analytics.json: per-speaker stats (display_name, utterance/word counts, participation, etc.).

## Configuration

- Qdrant URL: default http://localhost:6333
- GPU usage: enabled by default; automatically falls back to CPU
- Batch sizes: tuned for common GPUs; adjust if you encounter OOM or low utilization
- Base data path: set to your Conversation_Chunks directory before running

## Quick Start

1) Ensure Qdrant is running locally.
2) Configure the base data path to your Conversation_Chunks directory.
3) Run the main script:
```
bash
# Pass the base path as an argument (recommended)
python "src/__init__/Conversation Embedding Code.py" "D:/Path/To/Conversation_Chunks"

# Or run without an argument to use the script's default path
python "src/__init__/Conversation Embedding Code.py"
```
What happens:
- Ensures required Qdrant collections:
  - docling_conversations (main vectors, cosine distance)
  - processed_files_log (metadata-only tracking)
- Iterates through document subfolders:
  - Loads chunks and speaker analytics
  - Batch-encodes content and stores vectors with metadata
  - Builds speaker segments, encodes, and stores them
  - Marks files as processed to avoid reprocessing unchanged inputs
- Prints processed-file status and collection statistics

Reprocessing:
- You can force a re-run even if inputs are unchanged (e.g., by adjusting the `force_reprocess` parameter in the programmatic API).

## Programmatic Usage (Python)

The library surface supports:
- Creating collections
- Processing a directory or a single document
- Creating and storing speaker segments
- Searching by text query with optional metadata filters
- Retrieving processed-file status and collection stats

Example outline:
```
python
# Python
# Adjust import path to your project layout
from your_module import DoclingConversationEmbedder

embedder = DoclingConversationEmbedder(qdrant_url="http://localhost:6333", use_gpu=True)
embedder.create_collection()

# Process a directory of document subfolders
embedder.process_directory(r"/path/to/Conversation_Chunks", force_reprocess=False)

# Search examples
results = embedder.search_conversations("HR performance reviews", limit=3)
for r in results:
    text = r["payload"].get("text") or r["payload"].get("combined_text", "")
    print(r["score"], r["payload"].get("speaker_display_name"), text[:100])

# Collection statistics
print(embedder.get_collection_stats())

# Processed files status
for info in embedder.get_processed_files_status():
    print(info)
```
Note: Update the import path and base directory to match your environment.

## Qdrant Collections

- docling_conversations
  - Distance: cosine
  - Vector size: matches the embedding model dimension (e.g., 384 for all-MiniLM-L6-v2)
  - Payload: conversation IDs, chunk indices, text or combined_text, timestamps, speaker info, document metadata, and derived flags

- processed_files_log
  - Minimal vector used solely to store metadata about processed inputs
  - Payload: file_path, file_modified_time, document_name, processed_at, counts

## Performance Tips

- GPU memory
  - Reduce batch_size if you hit out-of-memory
  - Clear CUDA cache between large conversations if needed
- Throughput
  - Keep batches sizable for better GPU utilization
  - Avoid excessive per-item operations between batches

## Troubleshooting

- Cannot connect to Qdrant
  - Verify container/instance is running and ports are open
  - Confirm URL and port (6333) are correct
- CUDA not used
  - Verify torch.cuda.is_available() in a Python shell
  - Install a CUDA-enabled PyTorch matching your driver/CUDA version
- Vector size mismatch
  - Ensure your collection vector size matches your embedding model’s output dimension
  - If you change models, recreate the collection accordingly
- Nothing processes / everything “Already processed”
  - Processed-file tracking skips unchanged inputs; force a re-run if desired

## Roadmap Ideas

- Pluggable embedding models and vector sizes
- Advanced search filters (time ranges, speakers, document subsets)
- Optional RAG endpoints over the stored vectors
- Export/import utilities for collections

## Contributing

Issues and pull requests are welcome. Please include:
- Environment details (OS, Python, CUDA/GPU if applicable)
- Reproduction steps
- Logs and error messages (if any)

## License

Add your preferred license (e.g., MIT) and include a LICENSE file.
```
