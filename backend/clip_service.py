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

    def load_model(self):
        """Lazy load OpenCLIP model and tokenizer locally."""
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

    def extract_and_save_embeddings(self, video_dir: str, frames_dir: str, timestamps: list):
        """
        Generates vector embeddings for every frame in frames_dir,
        saves embeddings.npy and metadata.json in video_dir.
        """
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
                timestamp_val = timestamps[idx] if idx < len(timestamps) else idx * 1.0

                metadata.append({
                    "frame_index": idx + 1,
                    "filename": filename,
                    "timestamp": timestamp_val
                })
            except Exception as e:
                print(f"[OpenCLIP Warning] Failed to process frame {frame_path}: {e}")

        if not image_tensors:
            raise ValueError("Failed to preprocess frame images")

        # Stack into batch tensor
        batch_tensor = torch.stack(image_tensors).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(batch_tensor)
            # L2 Normalize embeddings
            image_features /= image_features.norm(dim=-1, keepdim=True)

        embeddings_np = image_features.cpu().numpy()

        # Save embeddings and metadata
        embeddings_path = os.path.join(video_dir, "embeddings.npy")
        metadata_path = os.path.join(video_dir, "metadata.json")

        np.save(embeddings_path, embeddings_np)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[OpenCLIP] Successfully generated {len(embeddings_np)} embeddings -> {embeddings_path}")
        return len(embeddings_np)

    def search(self, upload_base_dir: str, query_text: str, video_id: str = None, top_k: int = 6):
        """
        Encodes query_text into OpenCLIP text embedding, computes cosine similarity
        against saved frame embeddings, and returns ranked top_k search results.
        """
        self.load_model()

        if not query_text or not query_text.strip():
            return []

        # Encode text query
        text_tokens = self.tokenizer([query_text.strip()]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        text_embed_np = text_features.cpu().numpy().squeeze(0)

        # Collect candidate video directories
        video_dirs = []
        if video_id and os.path.exists(os.path.join(upload_base_dir, video_id)):
            video_dirs = [os.path.join(upload_base_dir, video_id)]
        else:
            video_dirs = [
                d for d in glob.glob(os.path.join(upload_base_dir, "*"))
                if os.path.isdir(d) and os.path.exists(os.path.join(d, "embeddings.npy"))
            ]

        all_results = []

        for v_dir in video_dirs:
            curr_video_id = os.path.basename(v_dir)
            embeddings_path = os.path.join(v_dir, "embeddings.npy")
            metadata_path = os.path.join(v_dir, "metadata.json")

            if not (os.path.exists(embeddings_path) and os.path.exists(metadata_path)):
                continue

            try:
                embeddings = np.load(embeddings_path)
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                # Compute cosine similarities (dot product of normalized vectors)
                similarities = np.dot(embeddings, text_embed_np)

                for idx, sim in enumerate(similarities):
                    meta = metadata[idx] if idx < len(metadata) else {}
                    filename = meta.get("filename", f"frame_{(idx+1):04d}.jpg")
                    timestamp = meta.get("timestamp", idx * 1.0)
                    
                    # Convert similarity score (typically 0.15 - 0.40 range in CLIP) into user-friendly percentage
                    score_val = float(sim)
                    percentage = round(max(0.0, min(100.0, ((score_val + 1.0) / 2.0) * 100)), 1)

                    all_results.append({
                        "video_id": curr_video_id,
                        "frame_index": meta.get("frame_index", idx + 1),
                        "filename": filename,
                        "frame_url": f"http://127.0.0.1:8000/uploads/{curr_video_id}/frames/{filename}",
                        "timestamp": timestamp,
                        "formatted_timestamp": self._format_timestamp(timestamp),
                        "raw_score": round(score_val, 4),
                        "similarity_percent": percentage
                    })
            except Exception as e:
                print(f"[OpenCLIP Search Error] Failed to search in {v_dir}: {e}")

        # Sort all results by raw similarity score descending
        all_results.sort(key=lambda x: x["raw_score"], reverse=True)

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

# Global service instance
clip_service = CLIPSearchService()
