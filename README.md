# Conversation Embedding (GPU) — README

This project ingests conversation data (chunks plus speaker analytics), generates embeddings using Sentence Transformers with optional GPU acceleration, and stores them in a Qdrant vector database. It also supports grouping contiguous utterances by the same speaker into segments and provides a basic semantic search over the stored content.

## Features

- GPU-accelerated embedding generation with memory monitoring (falls back to CPU if no GPU is available).
- Batch embedding for efficient processing.
- Stores conversation chunks and speaker segments in Qdrant with rich metadata (speaker info, timestamps, word counts, etc.).
- Tracks processed files to avoid reprocessing unchanged data.
- Simple semantic search API with optional metadata filters.
- Collection stats and processed-file status reporting.

## Project Structure

- src/
  - Conversation Embedding Code.py — main script and embedding pipeline
- Basic Validation Scripts/ — optional validations (if any)
- requirements.txt — Python dependencies

## Prerequisites

- Python 3.9+ recommended
- Qdrant running locally (default: http://localhost:6333)
  - Option 1: Docker
    - docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
  - Option 2: Native install (see Qdrant docs)
- Optional GPU (CUDA) for acceleration
  - A compatible NVIDIA GPU and CUDA toolkit/driver
  - A PyTorch build with CUDA support

## Installation

```shell script
# Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```


Dependencies (from requirements.txt):
- torch, torchvision, torchaudio
- sentence-transformers
- transformers, accelerate
- qdrant-client
- numpy

Note: If you need CUDA-enabled PyTorch, follow the installation matrix from the official PyTorch site to pick the right wheel for your CUDA version.

## Data Layout

Prepare a base directory of conversation documents. Each document is a folder that contains two JSON files:

- {document_name}_conversation_chunks.json
- {document_name}_speaker_analytics.json

Example layout:
- Conversation_Chunks/
  - ProjectKickoff/
    - ProjectKickoff_conversation_chunks.json
    - ProjectKickoff_speaker_analytics.json
  - PerformanceReview/
    - PerformanceReview_conversation_chunks.json
    - PerformanceReview_speaker_analytics.json

Expected content overview:
- conversation_chunks.json: Contains conversations keyed by conversation_id with chunk payloads (text, speaker, timestamp, chunk_index) and document_info.
- speaker_analytics.json: Contains per-speaker statistics (display_name, participation metrics, etc.).

Tip: The script expects to iterate through each subfolder inside the base directory and process matching pairs of files.

## Configuration

- Qdrant URL: Default http://localhost:6333. You can pass a different URL to the embedder when using the API from Python.
- GPU usage: Enabled by default; automatically falls back to CPU if CUDA isn’t available.
- Batch sizes: Defaults are tuned for a mid-range GPU and can be adjusted in the code if needed.

## Running the Pipeline

Quick start (runs end-to-end with sensible defaults):

```shell script
python "src/Conversation Embedding Code.py"
```


Before running:
- Ensure Qdrant is up (localhost:6333).
- Update the base_path inside the script to point to your Conversation_Chunks directory.

What it does:
1. Initializes the embedder (GPU if available).
2. Creates/ensures the required Qdrant collections:
   - docling_conversations (main vectors, cosine distance, dim=384 for all-MiniLM-L6-v2)
   - processed_files_log (tracking of processed input files)
3. Processes all document subfolders under base_path:
   - Loads chunks and speaker analytics
   - Generates batch embeddings for chunks
   - Stores points with rich payloads
   - Groups contiguous utterances per speaker into segments, embeds, and stores them
   - Marks files as processed to prevent reprocessing unchanged data
4. Prints processed-file status and collection statistics

Reprocessing:
- To force reprocessing (ignore processed-file checks), use the Python API and set force_reprocess=True for the relevant methods, or adjust the script accordingly.

## Programmatic Usage (Python)

Below is an outline of the main capabilities you can call from Python. These names are provided to help you locate the corresponding functions and are not full code listings.

- Initialization:
  - DoclingConversationEmbedder(qdrant_url="http://localhost:6333", use_gpu=True)

- Collections:
  - create_collection()

- Processing:
  - process_directory(base_path, force_reprocess=False)
  - process_conversation_file(base_path, document_name, force_reprocess=False)
  - create_and_store_segments(base_path, document_name, force_reprocess=False)

- Search:
  - search_conversations(query: str, limit: int = 5, filter_conditions: Optional[dict] = None) -> List[dict]

- Status/Stats:
  - get_processed_files_status() -> List[dict]
  - get_collection_stats() -> dict

Example (replace paths with yours):

```python
# Python
from src.Conversation_Embedding_Code import DoclingConversationEmbedder  # adjust import if needed

embedder = DoclingConversationEmbedder(qdrant_url="http://localhost:6333", use_gpu=True)
embedder.create_collection()
embedder.process_directory(r"/path/to/Conversation_Chunks")

results = embedder.search_conversations("HR performance reviews", limit=3)
for r in results:
    text = r["payload"].get("text") or r["payload"].get("combined_text", "")
    print(r["score"], r["payload"].get("speaker_display_name"), text[:100])
```


Note: The exact import path may vary depending on your environment and how you run the script. If using the file directly, you can run the script and modify the base path in the main section.

## Qdrant Collections

- docling_conversations
  - Vector size: 384 (Sentence Transformers all-MiniLM-L6-v2)
  - Distance: cosine
  - Payload includes conversation IDs, chunk indices, text, timestamps, speaker info, document metadata, and derived features.

- processed_files_log
  - Vector size: 1 (dummy vector — only metadata is used)
  - Tracks file path, file modification time, document name, processed timestamp, and counts.

## GPU Notes

- If a CUDA-capable GPU is available and use_gpu=True, the script will:
  - Print GPU name and total memory
  - Monitor and print GPU memory allocation over time
- If GPU is not available, the script uses CPU and prints a message accordingly.
- You can clear CUDA cache between large operations if needed.

## Troubleshooting

- Qdrant connection errors
  - Ensure Qdrant is running and reachable at the configured URL.
  - Check port conflicts (6333) and firewall rules.

- CUDA not used when expected
  - Confirm torch.cuda.is_available() returns True in a Python shell.
  - Install a CUDA-enabled PyTorch build matching your driver/CUDA version.

- Reprocessing doesn’t happen
  - The pipeline skips files that were already processed and unchanged.
  - Use force_reprocess=True in processing calls, or modify the script’s main section to force a re-run.

- Dimension mismatch in Qdrant
  - Ensure the embedding model matches the collection vector size (all-MiniLM-L6-v2 => 384).
  - If you change models, recreate the collection or create a new one with the correct size.

## License

Provide your chosen license here (e.g., MIT). If you haven’t decided, consider adding a LICENSE file.

## Acknowledgments

- Sentence Transformers (all-MiniLM-L6-v2) for embeddings
- Qdrant for vector storage and search

## Contributing

- Open issues or submit PRs for bug fixes and enhancements.
- Please include clear reproduction steps and environment details for any issues.

## Contact

For questions or support, please open an issue. If you prefer email or another channel, add contact details here.

— AI Assistant