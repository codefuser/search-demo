import os
import json
import glob
import time
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

PROMPT_TEMPLATES = [
    "a photo of {query}",
    "a video frame showing {query}",
    "a picture containing {query}",
    "{query}"
]

def extract_tensor_features(features_output):
    """Safely extracts PyTorch Tensor from HuggingFace output objects."""
    if isinstance(features_output, torch.Tensor):
        return features_output
    if hasattr(features_output, 'text_embeds') and features_output.text_embeds is not None:
        return features_output.text_embeds
    if hasattr(features_output, 'image_embeds') and features_output.image_embeds is not None:
        return features_output.image_embeds
    if hasattr(features_output, 'pooler_output') and features_output.pooler_output is not None:
        return features_output.pooler_output
    return features_output[0]


class CLIPSearchService:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._is_loaded = False
        
        # RAM cache for video frame embeddings
        self._embeddings_cache = {}
        
        # RAM cache for query text embeddings
        self._text_embed_cache = {}

    def load_model(self):
        """Lazy load CLIP model and processor ONCE into memory."""
        if self._is_loaded:
            return

        t0 = time.perf_counter()
        print(f"[MODEL] Loading CLIP model ('{self.model_name}' on {self.device})...")
        
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()
        self._is_loaded = True
        
        elapsed = time.perf_counter() - t0
        print(f"[MODEL] CLIP Model loaded successfully in {elapsed:.3f}s!")

    def get_or_load_cache(self, upload_base_dir: str, video_id: str):
        """Loads frame embeddings & metadata into RAM cache if not already present."""
        if video_id in self._embeddings_cache:
            return self._embeddings_cache[video_id]

        t0 = time.perf_counter()
        video_dir = os.path.join(upload_base_dir, video_id)
        embeddings_path = os.path.join(video_dir, "embeddings.npy")
        metadata_path = os.path.join(video_dir, "metadata.json")

        if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            try:
                embeddings_np = np.load(embeddings_path)
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                cache_entry = {
                    "embeddings": embeddings_np,
                    "metadata": metadata
                }
                self._embeddings_cache[video_id] = cache_entry
                elapsed = time.perf_counter() - t0
                print(f"[CACHE] Loaded {len(embeddings_np)} vectors into RAM for '{video_id}' in {elapsed:.3f}s.")
                return cache_entry
            except Exception as e:
                print(f"[CACHE ERROR] Failed to load cache for {video_id}: {e}")
                return None

        return None

    def extract_and_save_embeddings(self, video_dir: str, frames_dir: str, timestamps: list):
        """
        Generates vector embeddings for every frame image using CLIP.
        """
        video_id = os.path.basename(video_dir)
        embeddings_path = os.path.join(video_dir, "embeddings.npy")
        metadata_path = os.path.join(video_dir, "metadata.json")

        if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            print(f"[INDEXING] Existing embeddings found on disk for '{video_id}'. Skipping generation.")
            cached = self.get_or_load_cache(os.path.dirname(video_dir), video_id)
            if cached and len(cached["embeddings"]) > 0:
                return len(cached["embeddings"]), True

        self.load_model()

        t0_gen = time.perf_counter()
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
        if not frame_files:
            raise ValueError("No frame images found for embedding generation")

        print(f"[INDEXING] Generating CLIP embeddings for {len(frame_files)} frames...")

        images = []
        metadata = []

        for idx, frame_path in enumerate(frame_files):
            try:
                img = Image.open(frame_path).convert('RGB')
                images.append(img)

                filename = os.path.basename(frame_path)
                timestamp_val = timestamps[idx] if idx < len(timestamps) else float(idx)

                metadata.append({
                    "frame_index": idx + 1,
                    "filename": filename,
                    "timestamp": timestamp_val
                })
            except Exception as e:
                print(f"[CLIP Warning] Failed to process frame {frame_path}: {e}")

        if not images:
            raise ValueError("Failed to process frame images")

        batch_size = 32
        all_embeddings = []

        with torch.inference_mode():
            for i in range(0, len(images), batch_size):
                b_imgs = images[i:i + batch_size]
                inputs = self.processor(images=b_imgs, return_tensors="pt", padding=True).to(self.device)
                out = self.model.get_image_features(**inputs)
                image_features = extract_tensor_features(out)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                all_embeddings.append(image_features.cpu().numpy())

        embeddings_np = np.vstack(all_embeddings)
        gen_elapsed = time.perf_counter() - t0_gen
        print(f"[INDEXING] Generated {len(embeddings_np)} CLIP embeddings in {gen_elapsed:.3f}s.")

        # Save embeddings & metadata to disk
        np.save(embeddings_path, embeddings_np)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self._embeddings_cache[video_id] = {
            "embeddings": embeddings_np,
            "metadata": metadata
        }

        return len(embeddings_np), False

    def _get_ensembled_text_embedding(self, query_text: str) -> np.ndarray:
        """
        Generates ensembled & normalized text vector embedding across multiple prompt templates.
        """
        clean_query = query_text.strip().lower()
        if clean_query in self._text_embed_cache:
            return self._text_embed_cache[clean_query]

        prompts = [template.format(query=clean_query) for template in PROMPT_TEMPLATES]

        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            out = self.model.get_text_features(**inputs)
            text_features = extract_tensor_features(out)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            mean_feature = text_features.mean(dim=0, keepdim=True)
            mean_feature /= mean_feature.norm(dim=-1, keepdim=True)

        embed_np = mean_feature.cpu().numpy().squeeze(0)
        self._text_embed_cache[clean_query] = embed_np
        return embed_np

    def search(
        self,
        upload_base_dir: str,
        query_text: str,
        video_id: str = None,
        top_k: int = 20,
        similarity_threshold: float = 0.15
    ):
        """
        Guaranteed Reliable OpenCLIP Search Pipeline:
        1. Encodes search text query using prompt ensembling
        2. Computes matrix dot product cosine similarity against RAM cached frame embeddings
        3. Ranks top_k results descending (highest match first)
        4. Guarantees results are returned for indexed videos!
        """
        t_start_total = time.perf_counter()
        print(f"\n--- [SEARCH START] Query: '{query_text}' | Video ID: '{video_id or 'ALL'}' ---")

        self.load_model()

        if not query_text or not query_text.strip():
            return []

        # 1. Ensembled text embedding vector
        t0_text = time.perf_counter()
        target_embed = self._get_ensembled_text_embedding(query_text)
        text_elapsed = time.perf_counter() - t0_text

        # 2. Target video directories
        target_video_ids = []
        if video_id:
            target_video_ids = [video_id]
        else:
            target_video_ids = [
                os.path.basename(d) for d in glob.glob(os.path.join(upload_base_dir, "*"))
                if os.path.isdir(d) and os.path.exists(os.path.join(d, "embeddings.npy"))
            ]

        all_results = []

        # 3. Vector similarity calculation
        t0_sim = time.perf_counter()
        for vid in target_video_ids:
            cached_data = self.get_or_load_cache(upload_base_dir, vid)
            if not cached_data:
                continue

            embeddings = cached_data["embeddings"]
            metadata = cached_data["metadata"]

            # Matrix dot product cosine similarity
            similarities = np.dot(embeddings, target_embed)

            for idx, sim in enumerate(similarities):
                score_val = float(sim)
                meta = metadata[idx] if idx < len(metadata) else {}
                filename = meta.get("filename", f"frame_{(idx+1):04d}.jpg")
                timestamp = meta.get("timestamp", float(idx))
                percentage = round(max(0.0, min(100.0, ((score_val + 1.0) / 2.0) * 100)), 1)
                frame_image_url = f"http://127.0.0.1:8000/uploads/{vid}/frames/{filename}"

                all_results.append({
                    "similarity_score": round(score_val, 4),
                    "similarity_percent": percentage,
                    "timestamp": timestamp,
                    "formatted_timestamp": self._format_timestamp(timestamp),
                    "frame_image": frame_image_url,
                    "frame_index": meta.get("frame_index", idx + 1),
                    "filename": filename,
                    "video_id": vid
                })

        sim_elapsed = time.perf_counter() - t0_sim

        if not all_results:
            print(f"[SEARCH COMPLETE] No video frames found in {upload_base_dir}.\n")
            return []

        # 4. Sort descending by similarity score (highest match first)
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Filter items exceeding similarity_threshold if any exist; fallback to top items if all below threshold
        filtered_results = [item for item in all_results if item["similarity_score"] >= similarity_threshold]
        if not filtered_results:
            filtered_results = all_results[:top_k]

        top_results = filtered_results[:top_k]

        total_elapsed = time.perf_counter() - t_start_total
        print(f"--- [SEARCH COMPLETE] Returned Top {len(top_results)} matches in {total_elapsed:.4f}s (Text encode: {text_elapsed:.4f}s | Vector sim: {sim_elapsed:.4f}s) ---\n")

        return top_results

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        sec_int = int(seconds)
        hrs = sec_int // 3600
        mins = (sec_int % 3600) // 60
        secs = sec_int % 60
        if hrs > 0:
            return f"{hrs:02d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"

clip_service = CLIPSearchService()
