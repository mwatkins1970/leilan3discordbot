import os
import json
import logging
import aiohttp
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union
from username_normalizer import UsernameNormalizer

# Configuration
RESULTS_PER_CATEGORY = {
    'gpt': 4,
    'opus': 4,
    'essay': 2,
    'interview': 2
}
SIMILARITY_METHOD = "max"  # or "mean"
DEFAULT_TEMPLATE = "rag_template.txt"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Where to pull files from
HF_DATASET_BASE_URL = "https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main"

# Everything your code needs from that HF dataset:
REQUIRED_FILES = [
    # root-level
    "dialogue_embeddings_mpnet.npy",
    "essay_embeddings_mpnet.npy",
    "interview_embeddings_mpnet.npy",
    "dialogue_chunks_mpnet.json",
    "dialogue_metadata_mpnet.json",
    "essay_chunks_mpnet.json",
    "essay_metadata_mpnet.json",
    "interview_chunks_mpnet.json",
    "interview_metadata_mpnet.json",

    # subchunked folder
    "subchunked/dialogue_texts_subchunked.json",
    "subchunked/dialogue_metadata_subchunked.json",
    "subchunked/essay_chunks_mpnet.json",
    "subchunked/essay_metadata_mpnet.json",
    "subchunked/interview_chunks_mpnet.json",
    "subchunked/interview_metadata_mpnet.json",
]

class ChunkMetadata:
    def __init__(self, label: str):
        self.label = label
        self.type, self.subtype = self._parse_label(label)
    
    @staticmethod
    def _parse_label(label: str) -> Tuple[str, str]:
        if not label or '_' not in label:
            return '', ''
        prefix, suffix = label.split('_', 1)
        if prefix == 'gpt3':
            return 'gpt', suffix
        elif prefix == 'opus':
            return 'opus', suffix
        return '', ''

class SubchunkData:
    def __init__(self, subchunks: List[str], embeddings: np.ndarray, parent_indices: List[int]):
        self.subchunks = subchunks
        self.embeddings = embeddings
        self.parent_indices = parent_indices  # Maps each subchunk to its parent chunk index

class ContextRetriever:
    def __init__(self, embeddings_dir="embeddings"):
        """Initialize basic paths only. All heavy lifting is done in create()."""
        self.embeddings_dir = Path(embeddings_dir)
        self.subchunks_dir = self.embeddings_dir / "subchunked"

    @classmethod
    async def create(cls, embeddings_dir="embeddings"):
        """
        Async factory method to create and initialize a ContextRetriever instance.
        """
        retriever = cls(embeddings_dir)
        # Do async initialization here
        await retriever.ensure_embeddings_exist()
        
        # Load the sentence-transformers model
        retriever.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Load the data from disk
        retriever.load_data()
        
        return retriever

    async def ensure_embeddings_exist(self):
        """
        Check for each required file locally; if missing, download it from Hugging Face.
        """
        logger.info("Checking/Downloading RAG embedding files if needed...")
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.subchunks_dir.mkdir(parents=True, exist_ok=True)

        for rel_path in REQUIRED_FILES:
            local_path = self.embeddings_dir / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if not local_path.is_file():
                url = f"{HF_DATASET_BASE_URL}/{rel_path}"
                logger.info(f"Downloading: {rel_path} from {url}")
                await self.download_file(url, local_path)
                logger.info(f"Completed download of {rel_path}")
            else:
                logger.info(f"File {rel_path} already present locally. Skipping download.")

    async def download_file(self, url: str, dest_path: Path, chunk_size: int = 8192):
        """
        Helper to download a file via HTTP asynchronously, streaming to disk in chunk_size segments.
        """
        timeout = aiohttp.ClientTimeout(total=600)  # 10 minute timeout
        connector = aiohttp.TCPConnector(force_close=True)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                with open(dest_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)

    def load_data(self):
        """Load all necessary data files into memory, including subchunks."""
        try:
            logger.info("Loading data from local embedding files...")

            # dialogue data (for gpt + opus)
            self.dialogue_chunks = self.load_json(self.embeddings_dir / "dialogue_chunks_mpnet.json")
            dialogue_metadata_raw = self.load_json(self.embeddings_dir / "dialogue_metadata_mpnet.json")
            self.dialogue_metadata = [ChunkMetadata(label) for label in dialogue_metadata_raw]
            self.dialogue_subchunks = SubchunkData(
                subchunks=self.load_json(self.subchunks_dir / "dialogue_texts_subchunked.json"),
                embeddings=np.load(self.embeddings_dir / "dialogue_embeddings_mpnet.npy"),
                parent_indices=self.load_json(self.subchunks_dir / "dialogue_metadata_subchunked.json")
            )
            self.gpt_indices = [i for i, meta in enumerate(self.dialogue_metadata) if meta.type == 'gpt']
            self.opus_indices = [i for i, meta in enumerate(self.dialogue_metadata) if meta.type == 'opus']

            # essay data
            self.essay_chunks = self.load_json(self.embeddings_dir / "essay_chunks_mpnet.json")
            self.essay_metadata = self.load_json(self.embeddings_dir / "essay_metadata_mpnet.json")
            self.essay_subchunks = SubchunkData(
                subchunks=self.load_json(self.subchunks_dir / "essay_chunks_mpnet.json"),
                embeddings=np.load(self.embeddings_dir / "essay_embeddings_mpnet.npy"),
                parent_indices=self.load_json(self.subchunks_dir / "essay_metadata_mpnet.json")
            )

            # interview data
            self.interview_chunks = self.load_json(self.embeddings_dir / "interview_chunks_mpnet.json")
            self.interview_metadata = self.load_json(self.embeddings_dir / "interview_metadata_mpnet.json")
            self.interview_subchunks = SubchunkData(
                subchunks=self.load_json(self.subchunks_dir / "interview_chunks_mpnet.json"),
                embeddings=np.load(self.embeddings_dir / "interview_embeddings_mpnet.npy"),
                parent_indices=self.load_json(self.subchunks_dir / "interview_metadata_mpnet.json")
            )

            logger.info("Data loading complete.")
            logger.info(f"Dialogue chunks: {len(self.dialogue_chunks)}")
            logger.info(f"Dialogue subchunks: {len(self.dialogue_subchunks.subchunks)}")
            logger.info(f"Essay chunks: {len(self.essay_chunks)}")
            logger.info(f"Essay subchunks: {len(self.essay_subchunks.subchunks)}")
            logger.info(f"Interview chunks: {len(self.interview_chunks)}")
            logger.info(f"Interview subchunks: {len(self.interview_subchunks.subchunks)}")

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def load_json(self, path: Path) -> Any:
        """Helper to load JSON files."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for input text."""
        return self.model.encode(text, normalize_embeddings=True)

    async def retrieve_context_for_message(self, message_text: str) -> str:
        """
        Async method to retrieve context for a given message.
        """
        return await self.retrieve_contexts(message_text)

    async def retrieve_contexts(self, query: str) -> str:
        """
        Main method to retrieve and format relevant contexts. Returns a big string
        with sub-chunk references inserted in the appropriate sections.
        """
        normalizer = UsernameNormalizer()
        normalized_query = normalizer.normalize_message_history(query)
        print(f"\n{'='*80}\nProcessing query: {normalized_query}\n{'='*80}")

        query_embedding = self.get_embedding(normalized_query)
        print(f"\nGenerated query embedding with shape: {query_embedding.shape}")

        categories = {
            'gpt': self.dialogue_subchunks,
            'opus': self.dialogue_subchunks,
            'essay': self.essay_subchunks,
            'interview': self.interview_subchunks
        }

        filled_sections = {}
        for cat, subchunk_data in categories.items():
            print(f"\n{'='*40}\nProcessing {cat.upper()} category\n{'='*40}")
            results = self.get_filtered_chunks(cat, subchunk_data, query_embedding)
            print(f"\nDeduplicating {cat} results...")
            deduped = self.deduplicate_chunks(results, RESULTS_PER_CATEGORY[cat])
            print(f"Final {cat} results after deduplication: {len(deduped)}")
            filled_sections[cat] = "".join([
                self.format_chunk(text, meta, sim, idx)
                for text, meta, sim, idx in deduped
            ])

        template = self.get_template()
        for tag, content in filled_sections.items():
            template = template.replace(f"<{tag}>", content)
        return template

    def get_filtered_chunks(self, category: str, subchunk_data: SubchunkData,
                            query_embedding: np.ndarray) -> List[Tuple[str, Dict, float, int]]:
        """Get chunks filtered by category and sorted by similarity."""
        print(f"\nProcessing {category.upper()} chunks:")
        subchunk_sim = self.calculate_similarities(query_embedding, subchunk_data.embeddings)

        # Map subchunk similarities up to chunk-level
        chunk_sims = self.get_chunk_similarities(
            subchunk_sim, subchunk_data.parent_indices, SIMILARITY_METHOD
        )

        # Filter based on category (for gpt/opus)
        if category in ['gpt', 'opus']:
            valid_indices = self.gpt_indices if category == 'gpt' else self.opus_indices
            chunk_sims = {idx: sim for idx, sim in chunk_sims.items() if idx in valid_indices}

        # Turn into a sorted list
        results = []
        for chunk_idx, sim in chunk_sims.items():
            if category in ['gpt', 'opus']:
                chunk = self.dialogue_chunks[chunk_idx]
                meta = {'label': self.dialogue_metadata[chunk_idx].label}
            else:
                if category == 'essay':
                    chunk = self.essay_chunks[chunk_idx]
                    meta = self.essay_metadata[chunk_idx]
                else:
                    chunk = self.interview_chunks[chunk_idx]
                    meta = self.interview_metadata[chunk_idx]
            results.append((chunk, meta, sim, chunk_idx))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def calculate_similarities(self, target_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities between target and embeddings."""
        print(f"\nCalculating similarities:")
        print(f"Target embedding shape: {target_embedding.shape}")
        print(f"Embeddings matrix shape: {embeddings.shape}")
        similarities = np.dot(embeddings, target_embedding)
        print(f"Resulting similarities shape: {similarities.shape}")
        top_10 = sorted(similarities, reverse=True)[:10]
        print(f"Top 10 similarity scores: {top_10}")
        return similarities

    def get_chunk_similarities(self, subchunk_similarities: np.ndarray,
                               parent_indices: List[Dict],
                               method: str = "max") -> Dict[int, float]:
        """
        Group subchunk similarities by their parent chunk index,
        and compute a single value (max or mean).
        """
        chunk_sims = defaultdict(list)
        for i, sim in enumerate(subchunk_similarities):
            parent_idx = self.get_parent_index(parent_indices[i])
            chunk_sims[parent_idx].append(sim)

        result = {}
        for idx, sims in chunk_sims.items():
            if method == "max":
                result[idx] = max(sims)
            else:
                result[idx] = sum(sims) / len(sims)
        return result

    def get_parent_index(self, parent_data: Union[Dict, int]) -> int:
        """Extract the parent index from various metadata formats."""
        if isinstance(parent_data, dict):
            if 'original_chunk_index' in parent_data:
                return parent_data['original_chunk_index']
            elif 'qa_index' in parent_data:
                return parent_data['qa_index']
            else:
                raise KeyError(f"Unrecognized parent index format: {parent_data}")
        return parent_data

    def deduplicate_chunks(self, results: List[Tuple[str, Dict, float, int]],
                           max_results: int = 10) -> List[Tuple[str, Dict, float, int]]:
        """
        Remove near-duplicate or fully duplicated text from results, 
        then keep the top N by similarity.
        """
        final_results = []
        for chunk, meta, sim, idx in results:
            # quick check if chunk text is substring of anything already in final
            if any(chunk in r[0] or r[0] in chunk for r in final_results):
                logger.debug(f"Skipping duplicate chunk {idx}")
                continue
            final_results.append((chunk, meta, sim, idx))
            if len(final_results) >= max_results:
                break
        # sort again by similarity
        final_results.sort(key=lambda x: x[2], reverse=True)
        return final_results

    def format_chunk(self, text: str, metadata: Dict, similarity: float, index: int) -> str:
        """
        Format a single chunk with metadata for output in the final RAG prompt.
        """
        header = f"[segment index: {index}]\n[similarity: {similarity:.3f}]"
        if 'label' in metadata:
            chunk_meta = ChunkMetadata(metadata['label'])
            if chunk_meta.type == 'gpt':
                header += f"\n[GPT-3 model: {chunk_meta.subtype}]"
            elif chunk_meta.type == 'opus':
                header += f"\n[Opus voice: {chunk_meta.subtype}]"
        return f"{header}\n\n{text}\n\n{'-'*80}\n"

    def get_template(self) -> str:
        """Load or create the template string (the skeleton for RAG)."""
        template_path = Path(DEFAULT_TEMPLATE)
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        # fallback if no rag_template.txt
        return "<gpt>\n\n<opus>\n\n<essay>\n\n<interview>"