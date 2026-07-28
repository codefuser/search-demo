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

        # RAM cache for frame captions & captioning model
        self.caption_model = None
        self.caption_processor = None
        self._is_caption_loaded = False
        self._captions_cache = {}

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

    def load_caption_model(self):
        """
        Loads BLIP vision-language image captioning model lazily.
        """
        if self._is_caption_loaded:
            return

        t0 = time.perf_counter()
        print("[STAGE 1b - Vision Caption Model] Loading BLIP captioning model ('Salesforce/blip-image-captioning-base')...")
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
            self.caption_model.eval()
            self._is_caption_loaded = True
            elapsed = time.perf_counter() - t0
            print(f"[STAGE 1b - Vision Caption Model] Loaded successfully in {elapsed:.3f}s!")
        except Exception as e:
            print(f"[STAGE 1b - Vision Caption Model Warning] Failed to load BLIP model: {e}")

    def generate_caption_for_frame(self, frame_path: str) -> str:
        """
        Generates a natural language vision-language caption for a frame image.
        Caches captions in RAM so each frame is captioned only once.
        """
        if frame_path in self._captions_cache:
            return self._captions_cache[frame_path]

        if not os.path.exists(frame_path):
            return "Video frame preview"

        try:
            self.load_caption_model()
            if self.caption_model and self.caption_processor:
                raw_image = Image.open(frame_path).convert('RGB')
                inputs = self.caption_processor(raw_image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.caption_model.generate(**inputs, max_new_tokens=30)
                caption_text = self.caption_processor.decode(out[0], skip_special_tokens=True).strip()
                caption_text = caption_text.capitalize() if caption_text else "Video frame preview"
                self._captions_cache[frame_path] = caption_text
                return caption_text
        except Exception as e:
            print(f"[Caption Generation Warning] Could not caption {frame_path}: {e}")

        filename = os.path.basename(frame_path)
        fallback = f"Extracted frame ({filename})"
        self._captions_cache[frame_path] = fallback
        return fallback

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

    def search(
        self,
        upload_base_dir: str,
        query_text: str,
        video_id: str = None,
        top_k: int = 20,
        similarity_threshold: float = 0.25
    ):
        """
        Enhanced semantic retrieval pipeline:
        1. Query encoding via CLIP
        2. In-memory cosine similarity calculation
        3. Threshold filtering (similarity_score >= similarity_threshold)
        4. Vision-Language frame caption generation
        5. Caption-based second-stage reranking
        """
        t_start_total = time.perf_counter()
        print(f"\n--- [SEARCH START] Query: '{query_text}' | Video ID: '{video_id or 'ALL'}' | Threshold: {similarity_threshold} ---")

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
                score_val = float(sim)
                
                # Minimum Similarity Threshold Filter
                if score_val < similarity_threshold:
                    continue

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
        print(f"[STAGE 6 - Computing Similarities] Found {len(all_results)} frames above threshold {similarity_threshold} in {sim_elapsed:.4f}s.")

        if not all_results:
            print(f"[SEARCH COMPLETE] 0 frames exceeded similarity threshold {similarity_threshold}. Returning empty results.\n")
            return []

        # Sort by similarity score descending
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        candidates = all_results[:min(len(all_results), top_k * 2)]

        # Stage 7: Generate AI Captions & Perform Second-Stage Reranking
        t0_rerank = time.perf_counter()
        query_words = [w.lower() for w in query_text.strip().split() if len(w) > 2]

        for item in candidates:
            frame_local_path = os.path.join(upload_base_dir, item["video_id"], "frames", item["filename"])
            caption = self.generate_caption_for_frame(frame_local_path)
            item["caption"] = caption

            # Rerank boost if caption matches query words
            matches = sum(1 for w in query_words if w in caption.lower())
            if matches > 0:
                boost = round(matches * 0.04, 4)
                item["similarity_score"] = round(item["similarity_score"] + boost, 4)
                item["similarity_percent"] = round(max(0.0, min(100.0, ((item["similarity_score"] + 1.0) / 2.0) * 100)), 1)

        # Final sort after reranking
        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_results = candidates[:top_k]
        rerank_elapsed = time.perf_counter() - t0_rerank

        total_elapsed = time.perf_counter() - t_start_total
        print(f"[STAGE 7 - Captions & Reranking] Captioned & reranked Top {len(top_results)} results in {rerank_elapsed:.4f}s.")
        print(f"--- [SEARCH COMPLETE] Total search execution time: {total_elapsed:.4f}s ---\n")

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

