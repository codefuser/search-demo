import os
import json
import glob
import time
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPSearchService:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._is_loaded = False
        
        # RAM cache for video embeddings
        # Schema: { video_id: { "embeddings": np.ndarray, "metadata": list } }
        self._embeddings_cache = {}

    def load_model(self):
        """
        Loads CLIP model and processor ONCE at server startup.
        Subsequent calls return immediately without reloading.
        """
        if self._is_loaded:
            print(f"[STAGE 1 - Model Load] Model '{self.model_name}' is already loaded in memory (0.00s).")
            return

        t0 = time.perf_counter()
        print(f"[STAGE 1 - Model Load] Starting CLIP model loading ('{self.model_name}' on {self.device})...")
        
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()
        self._is_loaded = True
        
        elapsed = time.perf_counter() - t0
        print(f"[STAGE 1 - Model Load] COMPLETED in {elapsed:.3f}s")

    def get_or_load_cache(self, upload_base_dir: str, video_id: str):
        """Loads embeddings into RAM cache if not already present."""
        if video_id in self._embeddings_cache:
            return self._embeddings_cache[video_id]

        t0 = time.perf_counter()
        video_dir = os.path.join(upload_base_dir, video_id)
        embeddings_path = os.path.join(video_dir, "embeddings.npy")
        metadata_path = os.path.join(video_dir, "metadata.json")

        if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            try:
                print(f"[STAGE 5 - Loading Embeddings] Reading embeddings from disk for video_id '{video_id}'...")
                embeddings_np = np.load(embeddings_path)
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                cache_entry = {
                    "embeddings": embeddings_np,
                    "metadata": metadata
                }
                self._embeddings_cache[video_id] = cache_entry
                elapsed = time.perf_counter() - t0
                print(f"[STAGE 5 - Loading Embeddings] Loaded {len(embeddings_np)} vectors into RAM in {elapsed:.3f}s.")
                return cache_entry
            except Exception as e:
                print(f"[STAGE 5 - Loading Embeddings ERROR] Failed to load cache for {video_id}: {e}")
                return None

        return None

    def extract_and_save_embeddings(self, video_dir: str, frames_dir: str, timestamps: list):
        """
        Generates vector embeddings for every frame in frames_dir using CLIP.
        Skips regeneration if embeddings.npy already exists.
        """
        video_id = os.path.basename(video_dir)
        embeddings_path = os.path.join(video_dir, "embeddings.npy")
        metadata_path = os.path.join(video_dir, "metadata.json")

        # Optimization: Reuse existing embeddings
        if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            print(f"[STAGE 3 & 4] Existing embeddings found on disk for '{video_id}'. Skipping generation.")
            cached = self.get_or_load_cache(os.path.dirname(video_dir), video_id)
            if cached:
                return len(cached["embeddings"]), True

        self.load_model()

        t0_gen = time.perf_counter()
        print(f"[STAGE 3 - Generating Frame Embeddings] Processing frames from '{frames_dir}'...")

        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
        if not frame_files:
            raise ValueError("No frame images found for embedding generation")

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

        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            if hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
                image_features = image_features.pooler_output
            elif hasattr(image_features, "image_embeds") and image_features.image_embeds is not None:
                image_features = image_features.image_embeds
            image_features /= image_features.norm(dim=-1, keepdim=True)

        embeddings_np = image_features.cpu().numpy()
        gen_elapsed = time.perf_counter() - t0_gen
        print(f"[STAGE 3 - Generating Frame Embeddings] Generated {len(embeddings_np)} embeddings in {gen_elapsed:.3f}s.")

        # Stage 4: Save embeddings to disk
        t0_save = time.perf_counter()
        np.save(embeddings_path, embeddings_np)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        save_elapsed = time.perf_counter() - t0_save
        print(f"[STAGE 4 - Saving Embeddings] Saved to disk ({embeddings_path}) in {save_elapsed:.3f}s.")

        self._embeddings_cache[video_id] = {
            "embeddings": embeddings_np,
            "metadata": metadata
        }

        return len(embeddings_np), False

    def search(self, upload_base_dir: str, query_text: str, video_id: str = None, top_k: int = 20):
        """
        Fast non-blocking semantic search pipeline:
        1. Load model if not loaded (0.00s if pre-loaded at startup)
        2. Generate text embedding for user query ONLY
        3. Load cached frame embeddings from RAM
        4. Compute vector dot product similarity
        5. Return ranked top_k results
        """
        t_start_total = time.perf_counter()
        print(f"\n--- [SEARCH START] Query: '{query_text}' | Video ID: '{video_id or 'ALL'}' ---")

        # Stage 1: Verify model loaded
        self.load_model()

        if not query_text or not query_text.strip():
            return []

        # Stage 6a: Generate text embedding vector
        t0_text = time.perf_counter()
        print(f"[STAGE 6 - Text Embedding] Encoding text query '{query_text}'...")
        inputs = self.processor(text=[query_text.strip()], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            if hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
                text_features = text_features.pooler_output
            elif hasattr(text_features, "text_embeds") and text_features.text_embeds is not None:
                text_features = text_features.text_embeds
            text_features /= text_features.norm(dim=-1, keepdim=True)

        text_embed_np = text_features.cpu().numpy().squeeze(0)
        text_elapsed = time.perf_counter() - t0_text
        print(f"[STAGE 6 - Text Embedding] Encoded text vector in {text_elapsed:.4f}s.")

        # Stage 5: Retrieve cached frame embeddings
        t0_cache = time.perf_counter()
        target_video_ids = []
        if video_id:
            target_video_ids = [video_id]
        else:
            target_video_ids = [
                os.path.basename(d) for d in glob.glob(os.path.join(upload_base_dir, "*"))
                if os.path.isdir(d) and os.path.exists(os.path.join(d, "embeddings.npy"))
            ]

        all_results = []

        # Stage 6b: Compute matrix multiplication / similarity dot product
        t0_sim = time.perf_counter()
        for vid in target_video_ids:
            cached_data = self.get_or_load_cache(upload_base_dir, vid)
            if not cached_data:
                continue

            embeddings = cached_data["embeddings"]
            metadata = cached_data["metadata"]

            # Pure in-memory vector comparison
            similarities = np.dot(embeddings, text_embed_np)

            for idx, sim in enumerate(similarities):
                meta = metadata[idx] if idx < len(metadata) else {}
                filename = meta.get("filename", f"frame_{(idx+1):04d}.jpg")
                timestamp = meta.get("timestamp", float(idx))
                
                score_val = float(sim)
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
        print(f"[STAGE 6 - Computing Similarities] Calculated dot product across {len(all_results)} frames in {sim_elapsed:.4f}s.")

        # Stage 7: Sort and return top_k results
        t0_sort = time.perf_counter()
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_results = all_results[:top_k]
        sort_elapsed = time.perf_counter() - t0_sort

        total_elapsed = time.perf_counter() - t_start_total
        print(f"[STAGE 7 - Returning Results] Prepared Top {len(top_results)} matches in {sort_elapsed:.4f}s.")
        print(f"--- [SEARCH COMPLETE] Total search pipeline execution time: {total_elapsed:.4f}s ---\n")

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
