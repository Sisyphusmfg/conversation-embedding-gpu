import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional
import torch
import hashlib
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from datetime import datetime


class DoclingConversationEmbedder:
    def __init__(self, qdrant_url: str = "http://localhost:6333", use_gpu: bool = True):
        """Initialize embedder for docling-processed conversation data."""
        self.client = QdrantClient(url=qdrant_url)

        # GPU configuration
        self.device = self._setup_device(use_gpu)
        print(f"Using device: {self.device}")

        # Load model with GPU support
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        # Collection names
        self.collection_name = "docling_conversations"
        self.processed_files_collection = "processed_files_log"
        self.vector_size = 384

    def _setup_device(self, use_gpu: bool) -> str:
        """Setup computing device (GPU/CPU)."""
        if use_gpu and torch.cuda.is_available():
            device = "cuda"
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = "cpu"
            if use_gpu:
                print("GPU requested but not available, using CPU")
            else:
                print("Using CPU as requested")
        return device

    def monitor_gpu_memory(self, context: str = ""):
        """Monitor GPU memory usage with context."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            print(
                f"GPU Memory {context}- Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Peak: {max_allocated:.2f}GB")
        else:
            print(f"GPU Memory {context}- Not using GPU")

    def get_file_modification_time(self, file_path: str) -> str:
        """Get file modification time as ISO string."""
        try:
            file_obj = Path(file_path)
            mod_time = file_obj.stat().st_mtime
            return datetime.fromtimestamp(mod_time).isoformat()
        except Exception as e:
            print(f"Error getting file modification time: {e}")
            return datetime.now().isoformat()

    def create_collection(self):
        """Create Qdrant collection for storing conversation embeddings."""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
        except Exception as e:
            print(f"Collection might already exist: {e}")

        # Create collection for tracking processed files
        try:
            self.client.create_collection(
                collection_name=self.processed_files_collection,
                vectors_config=models.VectorParams(
                    size=1,  # Minimal vector, we only need metadata
                    distance=models.Distance.COSINE
                )
            )
        except Exception as e:
            print(f"Processed files collection might already exist: {e}")

    def is_file_processed(self, file_path: str, file_modified_time: str) -> bool:
        """Check if a file has already been processed based on path and modification time."""
        try:
            # Use scroll with filter instead of vector search for efficiency
            scroll_result = self.client.scroll(
                collection_name=self.processed_files_collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchValue(value=file_path)
                        )
                    ]
                ),
                limit=100,  # Allow multiple records per file
                with_payload=True
            )

            points = scroll_result[0]

            # Check all matching records for the modification time
            for point in points:
                stored_mod_time = point.payload.get("file_modified_time")
                if stored_mod_time == file_modified_time:
                    return True

            return False
        except Exception as e:
            print(f"Error checking file processing status: {e}")
            return False

    def mark_file_processed(self, file_path: str, file_modified_time: str,
                            document_name: str, chunk_count: int):
        """Mark a file as processed in the tracking collection."""
        try:
            # Generate deterministic UUID from file path using SHA256
            path_hash = hashlib.sha256(file_path.encode('utf-8')).hexdigest()
            # Create UUID from first 32 chars of hash (deterministic but valid UUID format)
            point_id = str(uuid.UUID(path_hash[:32]))

            point = models.PointStruct(
                id=point_id,  # Deterministic UUID prevents duplicates
                vector=[0.0],  # Dummy vector
                payload={
                    "file_path": file_path,
                    "file_modified_time": file_modified_time,
                    "document_name": document_name,
                    "processed_at": datetime.now().isoformat(),
                    "chunk_count": chunk_count
                }
            )

            self.client.upsert(
                collection_name=self.processed_files_collection,
                points=[point]
            )
        except Exception as e:
            print(f"Error marking file as processed: {e}")

    def load_conversation_data(self, conversation_chunks_path: str) -> Dict:
        """Load conversation chunks from docling JSON file."""
        with open(conversation_chunks_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_speaker_analytics(self, speaker_analytics_path: str) -> Dict:
        """Load speaker analytics from docling JSON file."""
        with open(speaker_analytics_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_embeddings(self, text: str) -> List[float]:
        """Generate vector embeddings for text."""
        embedding = self.model.encode(text, convert_to_tensor=False, show_progress_bar=False)
        return embedding.tolist()

    def create_embeddings_batch(self, texts: List[str], batch_size: int = 48) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches for GPU efficiency."""
        self.monitor_gpu_memory("before batch embedding")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=False,
            show_progress_bar=True if len(texts) > 50 else False
        )

        self.monitor_gpu_memory("after batch embedding")
        return embeddings.tolist()

    def store_conversation_chunks(self, conversation_id: str, chunks: List[Dict],
                                  document_info: Dict, speaker_analytics: Dict):
        """Store individual conversation chunks as vectors in Qdrant with GPU-optimized batch processing."""
        # Extract all texts for batch processing
        texts = [chunk["payload"]["text"] for chunk in chunks]

        # Generate embeddings in batch for GPU efficiency
        print(f"Generating {len(texts)} embeddings for conversation {conversation_id}...")
        embeddings = self.create_embeddings_batch(texts, batch_size=48)  # Optimized for GTX 1080

        points = []

        for i, chunk in enumerate(chunks):
            payload = chunk["payload"]
            extended_metadata = chunk.get("extended_metadata", {})

            # Use pre-computed embedding
            embedding = embeddings[i]

            # Extract speaker information
            speaker_id = payload["speaker"]
            speaker_info = speaker_analytics["speakers"].get(speaker_id, {})

            # Get schema version info from chunk or document
            schema_version = extended_metadata.get("schema_version", document_info.get("schema_version", "unknown"))
            tool_version = extended_metadata.get("tool_version", document_info.get("tool_version", "unknown"))

            # Create comprehensive payload with enhanced metadata
            point_payload = {
                # Core conversation data
                "conversation_id": conversation_id,
                "chunk_index": payload["chunk_index"],
                "text": payload["text"],
                "timestamp": payload["timestamp"],
                "speaker_id": speaker_id,

                # Document metadata
                "document_name": document_info["document_name"],
                "source_file": document_info["source_file"],
                "total_chunks": document_info["total_chunks"],
                "processed_at": document_info["processed_at"],
                "schema_version": schema_version,
                "tool_version": tool_version,

                # Enhanced chunk-specific metadata
                "chunk_method": extended_metadata.get("metadata", {}).get("chunk_method", "unknown"),
                "estimated_duration_seconds": extended_metadata.get("metadata", {}).get("estimated_duration_seconds",
                                                                                        0),
                "text_length": len(payload["text"]),
                "word_count": len(payload["text"].split()),

                # Enhanced speaker information with new analytics
                "speaker_display_name": speaker_info.get("display_name", "Unknown"),
                "speaker_type": speaker_info.get("speaker_type", "unknown"),
                "speaker_utterance_count": speaker_info.get("utterance_count", 0),
                "speaker_total_words": speaker_info.get("total_words", 0),
                "speaker_participation_percentage": speaker_info.get("participation_percentage", 0),
                "speaker_word_percentage": speaker_info.get("word_percentage", 0),
                "speaker_avg_words_per_utterance": speaker_info.get("average_words_per_utterance", 0),

                # Extended speaker metadata
                "utterance_id": extended_metadata.get("speaker_metadata", {}).get("utterance_id", ""),
                "original_speaker_label": extended_metadata.get("speaker_metadata", {}).get("original_speaker_label",
                                                                                            ""),

                # Content analysis
                "is_short_utterance": len(payload["text"].split()) < 10,
                "is_long_utterance": len(payload["text"].split()) > 50,
                "contains_question": "?" in payload["text"],

                # Timing and flow analysis
                "has_timing_estimate": extended_metadata.get("metadata", {}).get("estimated_duration_seconds", 0) > 0,
                "processing_method": document_info.get("processing_method", "unknown"),
            }

            # Create point
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=point_payload
            )
            points.append(point)

        # Batch upload to Qdrant
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Stored {len(points)} chunks for conversation {conversation_id}")

    def create_speaker_segments(self, conversation_id: str, chunks: List[Dict],
                                speaker_analytics: Dict) -> List[Dict]:
        """Group consecutive chunks by the same speaker into segments."""
        segments = []
        current_segment = []
        current_speaker = None

        for chunk in chunks:
            speaker = chunk["payload"]["speaker"]

            if speaker != current_speaker:
                # Save previous segment if it exists
                if current_segment:
                    segments.append({
                        "speaker": current_speaker,
                        "chunks": current_segment,
                        "combined_text": " ".join([c["payload"]["text"] for c in current_segment]),
                        "start_timestamp": current_segment[0]["payload"]["timestamp"],
                        "end_timestamp": current_segment[-1]["payload"]["timestamp"],
                        "chunk_count": len(current_segment)
                    })

                # Start new segment
                current_segment = [chunk]
                current_speaker = speaker
            else:
                current_segment.append(chunk)

        # Don't forget the last segment
        if current_segment:
            segments.append({
                "speaker": current_speaker,
                "chunks": current_segment,
                "combined_text": " ".join([c["payload"]["text"] for c in current_segment]),
                "start_timestamp": current_segment[0]["payload"]["timestamp"],
                "end_timestamp": current_segment[-1]["payload"]["timestamp"],
                "chunk_count": len(current_segment)
            })

        return segments

    def store_speaker_segments(self, conversation_id: str, segments: List[Dict],
                               document_info: Dict, speaker_analytics: Dict):
        """Store speaker segments as vectors in Qdrant with GPU batch processing."""
        # Extract texts for batch processing
        texts = [segment["combined_text"] for segment in segments]

        # Generate embeddings in batch
        if texts:
            embeddings = self.create_embeddings_batch(texts, batch_size=32)
        else:
            embeddings = []

        points = []

        for i, segment in enumerate(segments):
            # Use pre-computed embedding
            embedding = embeddings[i] if i < len(embeddings) else self.create_embeddings(segment["combined_text"])

            speaker_info = speaker_analytics["speakers"].get(segment["speaker"], {})

            point_payload = {
                "conversation_id": conversation_id,
                "segment_id": f"{conversation_id}_segment_{i}",
                "segment_type": "speaker_segment",
                "speaker_id": segment["speaker"],
                "speaker_display_name": speaker_info.get("display_name", "Unknown"),
                "combined_text": segment["combined_text"],
                "chunk_count": segment["chunk_count"],
                "start_timestamp": segment["start_timestamp"],
                "end_timestamp": segment["end_timestamp"],
                "word_count": len(segment["combined_text"].split()),
                "document_name": document_info["document_name"],
                "source_file": document_info["source_file"],
                "speaker_participation_percentage": speaker_info.get("participation_percentage", 0),
            }

            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=point_payload
            )
            points.append(point)

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Stored {len(points)} speaker segments for conversation {conversation_id}")

    def validate_chunk_format(self, chunk: Dict) -> bool:
        """Validate that chunk has the expected format from the new chunking system."""
        try:
            # Check for required top-level structure
            if "payload" not in chunk:
                return False

            payload = chunk["payload"]
            required_fields = ["conversation_id", "chunk_index", "text", "timestamp", "speaker"]

            for field in required_fields:
                if field not in payload:
                    return False

            # Check for extended metadata (optional but expected from new system)
            if "extended_metadata" in chunk:
                metadata = chunk["extended_metadata"]
                if "schema_version" in metadata and "tool_version" in metadata:
                    # New format detected
                    return True

            # Backward compatibility - old format is still valid
            return True

        except Exception as e:
            print(f"Error validating chunk format: {e}")
            return False

    def process_conversation_file(self, base_path: str, document_name: str, force_reprocess: bool = False):
        """Process a complete document folder with all its conversation data."""
        doc_folder = Path(base_path) / document_name

        # Load main files
        chunks_file = doc_folder / f"{document_name}_conversation_chunks.json"
        analytics_file = doc_folder / f"{document_name}_speaker_analytics.json"

        if not chunks_file.exists() or not analytics_file.exists():
            print(f"Missing files for document {document_name}")
            return

        # Check if file was already processed (unless forcing reprocess)
        if not force_reprocess:
            file_mod_time = self.get_file_modification_time(str(chunks_file))
            if self.is_file_processed(str(chunks_file), file_mod_time):
                print(f"File {document_name} already processed and up to date. Skipping...")
                return

        print(f"Starting processing for document: {document_name}")
        self.monitor_gpu_memory("at start of document processing")

        # Load data
        conversation_data = self.load_conversation_data(str(chunks_file))
        speaker_analytics = self.load_speaker_analytics(str(analytics_file))

        # Validate chunk format compatibility
        sample_chunk = None
        for conv_id, chunks in conversation_data["conversations"].items():
            if chunks:
                sample_chunk = chunks[0]
                break

        if sample_chunk and not self.validate_chunk_format(sample_chunk):
            print(f"Warning: Chunk format validation failed for {document_name}. Proceeding with best effort.")

        total_chunks = 0

        # Process each conversation in the document
        for conv_id, chunks in conversation_data["conversations"].items():
            self.store_conversation_chunks(
                conv_id,
                chunks,
                conversation_data["document_info"],
                speaker_analytics
            )
            total_chunks += len(chunks)

            # Monitor memory after each conversation
            self.monitor_gpu_memory(f"after processing conversation {conv_id}")

            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Mark file as processed
        file_mod_time = self.get_file_modification_time(str(chunks_file))
        self.mark_file_processed(str(chunks_file), file_mod_time, document_name, total_chunks)

        print(f"Completed document: {document_name} ({total_chunks} chunks)")
        self.monitor_gpu_memory("at end of document processing")

    def create_and_store_segments(self, base_path: str, document_name: str, force_reprocess: bool = False):
        """Create and store speaker segments for a document."""
        doc_folder = Path(base_path) / document_name
        chunks_file = doc_folder / f"{document_name}_conversation_chunks.json"
        analytics_file = doc_folder / f"{document_name}_speaker_analytics.json"

        # Skip if already processed (unless forcing)
        if not force_reprocess:
            # Check modification times of BOTH files
            chunks_mod_time = self.get_file_modification_time(str(chunks_file))
            analytics_mod_time = self.get_file_modification_time(str(analytics_file))

            # Create a combined modification signature
            combined_mod_time = f"{chunks_mod_time}|{analytics_mod_time}"

            if self.is_file_processed(f"{str(chunks_file)}_segments", combined_mod_time):
                return

        conversation_data = self.load_conversation_data(str(chunks_file))
        speaker_analytics = self.load_speaker_analytics(str(analytics_file))

        total_segments = 0
        for conv_id, chunks in conversation_data["conversations"].items():
            segments = self.create_speaker_segments(conv_id, chunks, speaker_analytics)
            self.store_speaker_segments(
                conv_id,
                segments,
                conversation_data["document_info"],
                speaker_analytics
            )
            total_segments += len(segments)

        # Mark segments as processed with combined modification time
        chunks_mod_time = self.get_file_modification_time(str(chunks_file))
        analytics_mod_time = self.get_file_modification_time(str(analytics_file))
        combined_mod_time = f"{chunks_mod_time}|{analytics_mod_time}"

        self.mark_file_processed(f"{str(chunks_file)}_segments", combined_mod_time, f"{document_name}_segments",
                                 total_segments)

    def process_directory(self, base_path: str, force_reprocess: bool = False):
        """Process all documents in the conversation chunks directory."""
        base_path = Path(base_path)

        if not base_path.exists():
            print(f"Directory {base_path} does not exist")
            return

        # Find all document folders
        processed_count = 0
        skipped_count = 0

        for item in base_path.iterdir():
            if item.is_dir():
                document_name = item.name
                print(f"Processing document: {document_name}")

                try:
                    # Check if already processed before attempting
                    chunks_file = item / f"{document_name}_conversation_chunks.json"
                    if chunks_file.exists() and not force_reprocess:
                        file_mod_time = self.get_file_modification_time(str(chunks_file))
                        if self.is_file_processed(str(chunks_file), file_mod_time):
                            print(f"  -> Already processed, skipping...")
                            skipped_count += 1
                            continue

                    self.process_conversation_file(str(base_path), document_name, force_reprocess)

                    # Also create speaker segments
                    self.create_and_store_segments(str(base_path), document_name, force_reprocess)

                    processed_count += 1

                except Exception as e:
                    print(f"Error processing {document_name}: {e}")

        print(f"\nProcessing complete: {processed_count} processed, {skipped_count} skipped")

    def search_conversations(self, query: str, limit: int = 5,
                             filter_conditions: Optional[Dict] = None) -> List[Dict]:
        """Search for similar conversation content."""
        query_embedding = self.create_embeddings(query)

        # Build filter if provided
        query_filter = None
        if filter_conditions:
            must_conditions = []
            for key, value in filter_conditions.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            if must_conditions:
                query_filter = models.Filter(must=must_conditions)

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )

        results = []
        for point in search_result.points:
            results.append({
                'score': point.score,
                'payload': point.payload
            })

        return results

    def get_processed_files_status(self) -> List[Dict]:
        """Get list of all processed files and their status."""
        try:
            result = self.client.scroll(
                collection_name=self.processed_files_collection,
                limit=1000,
                with_payload=True
            )

            files_status = []
            for point in result[0]:
                files_status.append(point.payload)

            return files_status
        except Exception as e:
            print(f"Error getting processed files status: {e}")
            return []

    def get_collection_stats(self) -> Dict:
        """Get statistics about the stored conversation data."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "total_vectors": info.vectors_count,
                "indexed_vectors": info.indexed_vectors_count,
                "collection_status": info.status
            }
        except Exception as e:
            return {"error": str(e)}


# Example usage with GPU optimization and memory monitoring
def main():
    import sys

    # Check for command line argument
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
        print(f"Using provided path: {base_path}")
    else:
        base_path = r"D:\Docling Trial\Conversation_Chunks"
        print(f"Using default path: {base_path}")

    # Initialize embedder with GPU support
    embedder = DoclingConversationEmbedder(use_gpu=True)

    # Monitor initial GPU state
    embedder.monitor_gpu_memory("at startup")

    # Create collection
    embedder.create_collection()

    # Remove the old hardcoded path line and use the dynamic path
    # Process your conversation data directory
    # base_path = r"D:\Docling Trial\Conversation_Chunks"  # OLD LINE - REMOVED

    # Process with GPU acceleration and monitoring
    print("=== GPU-Accelerated Processing with Memory Monitoring ===")
    embedder.process_directory(base_path)

    # Final memory check
    embedder.monitor_gpu_memory("after all processing")

    # Show processed files status
    print("\n=== Processed Files Status ===")
    processed_files = embedder.get_processed_files_status()
    for file_info in processed_files:
        print(f"File: {file_info['document_name']}")
        print(f"  Processed: {file_info['processed_at']}")
        print(f"  Chunks: {file_info['chunk_count']}")
        print("---")

    # Example searches
    print("\n=== Search Examples ===")

    # Search for HR-related content
    hr_results = embedder.search_conversations("HR performance reviews", limit=3)
    print("\nHR Performance Reviews:")
    for result in hr_results:
        print(f"Score: {result['score']:.4f}")
        print(f"Speaker: {result['payload']['speaker_display_name']}")

        # Handle both individual chunks and speaker segments
        text_content = result['payload'].get('text') or result['payload'].get('combined_text', '')
        print(f"Text: {text_content[:100]}...")
        print("---")

    # Get collection statistics
    stats = embedder.get_collection_stats()
    print(f"\nCollection Stats: {stats}")

    # GPU memory cleanup and final report
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        embedder.monitor_gpu_memory("after cleanup")

        # Reset peak memory counter for next run
        torch.cuda.reset_peak_memory_stats()
        print("GPU memory cache cleared and stats reset")


if __name__ == "__main__":
    main()