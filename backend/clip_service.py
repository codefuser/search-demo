import os
import json
import glob
import torch
import numpy as np
from PIL import Image
import open_clip

class CLIPSearchService:
    def __init__(self, model_name: str = 'ViT-B-32', pretrained: str = 'laion2b_s34b_b79k'):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._is_loaded = False
        
        # In-memory RAM cache for loaded embeddings and metadata
        # Schema: { video_id: { "embeddings": np.ndarray, "metadata": list } }
        self._embeddings_cache = {}

    def load_model(self):
        """Lazy load OpenCLIP model and tokenizer once into memory."""
        if not self._is_loaded:
            print(f"[OpenCLIP] Loading model '{self.model_name}' ({self.pretrained}) on device '{self.device}'...")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            self.model.eval()
            self._is_loaded = True
            print("[OpenCLIP] Model successfully loaded into memory.")

    def get_or_load_cache(self, upload_base_dir: str, video_id: str):
        """
        Loads embeddings and metadata into in-memory cache ONLY ONCE per video_id.
        Subsequent calls retrieve vectors directly from RAM.
        """
        if video_id in self._embeddings_cache:
            return self._embeddings_cache[video_id]

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
                print(f"[RAM Cache] Loaded {len(embeddings_np)} embeddings into RAM for video_id '{video_id}'.")
                return cache_entry
            except Exception as e:
                print(f"[RAM Cache Error] Failed to load cache for {video_id}: {e}")
                return None

        return None

    def extract_and_save_embeddings(self, video_dir: str, frames_dir: str, timestamps: list):
        """
        Generates vector embeddings for every frame in frames_dir using OpenCLIP.
        OPTIMIZATION: Does NOT regenerate embeddings if embeddings.npy already exists.
        Caches embeddings directly into RAM memory.
        """
        video_id = os.path.basename(video_dir)
        embeddings_path = os.path.join(video_dir, "embeddings.npy")
        metadata_path = os.path.join(video_dir, "metadata.json")

        # Optimization: Do not regenerate embeddings if they already exist on disk
        if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            print(f"[OpenCLIP Cache] Existing embeddings found for '{video_id}'. Skipping generation.")
            cached = self.get_or_load_cache(os.path.dirname(video_dir), video_id)
            if cached:
                return len(cached["embeddings"]), True  # (count, is_from_cache=True)

        self.load_model()

        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
        if not frame_files:
            raise ValueError("No frame images found for embedding generation")

        image_tensors = []
        metadata = []

        for idx, frame_path in enumerate(frame_files):
            try:
                img = Image.open(frame_path).convert('RGB')
                processed_img = self.preprocess(img)
                image_tensors.append(processed_img)

                filename = os.path.basename(frame_path)
                timestamp_val = timestamps[idx] if idx < len(timestamps) else float(idx)

                metadata.append({
                    "frame_index": idx + 1,
                    "filename": filename,
                    "timestamp": timestamp_val
                })
            except Exception as e:
                print(f"[OpenCLIP Warning] Failed to process frame {frame_path}: {e}")

        if not image_tensors:
            raise ValueError("Failed to preprocess frame images")

        batch_tensor = torch.stack(image_tensors).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(batch_tensor)
            # L2 Normalize frame embeddings
            image_features /= image_features.norm(dim=-1, keepdim=True)

        embeddings_np = image_features.cpu().numpy()

        # Save to disk
        np.save(embeddings_path, embeddings_np)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Store in RAM cache
        self._embeddings_cache[video_id] = {
            "embeddings": embeddings_np,
            "metadata": metadata
        }

        print(f"[OpenCLIP] Generated and cached {len(embeddings_np)} embeddings -> {embeddings_path}")
        return len(embeddings_np), False  # (count, is_from_cache=False)

    def search(self, upload_base_dir: str, query_text: str, video_id: str = None, top_k: int = 10):
        """
        Fast in-memory semantic search:
        - Encodes query text into embedding vector
        - Uses cached in-memory numpy matrix multiplication (np.dot)
        - Zero disk re-reading during search
        """
        self.load_model()

        if not query_text or not query_text.strip():
            return []

        # Generate text embedding
        text_tokens = self.tokenizer([query_text.strip()]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        text_embed_np = text_features.cpu().numpy().squeeze(0)

        # Retrieve video IDs
        target_video_ids = []
        if video_id:
            target_video_ids = [video_id]
        else:
            target_video_ids = [
                os.path.basename(d) for d in glob.glob(os.path.join(upload_base_dir, "*"))
                if os.path.isdir(d) and os.path.exists(os.path.join(d, "embeddings.npy"))
            ]

        all_results = []

        # Vector comparison against cached RAM embeddings
        for vid in target_video_ids:
            cached_data = self.get_or_load_cache(upload_base_dir, vid)
            if not cached_data:
                continue

            embeddings = cached_data["embeddings"]
            metadata = cached_data["metadata"]

            # Matrix multiplication / dot product in memory
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

        # Sort descending by similarity score (highest match first)
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)

        return all_results[:top_k]

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        sec_int = int(seconds)
        hrs = sec_int // 3600
        mins = (sec_int % 3600) // 60
        secs = sec_int % 60
        if hrs > 0:
            return f"{hrs:02d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"

# Singleton service instance
clip_service = CLIPSearchService()
