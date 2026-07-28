import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { VideoUploader } from './components/VideoUploader';
import { VideoPreview } from './components/VideoPreview';
import { SearchBar } from './components/SearchBar';
import { ResultsSection } from './components/ResultsSection';

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadStatus, setUploadStatus] = useState<{
    type: 'idle' | 'success' | 'error';
    message: string;
  }>({ type: 'idle', message: '' });

  const handleVideoSelect = (file: File) => {
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setUploadStatus({ type: 'idle', message: '' });
  };

  const handleClearVideo = () => {
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }
    setSelectedFile(null);
    setVideoUrl(null);
    setUploadStatus({ type: 'idle', message: '' });
  };

  const handleUploadToBackend = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setUploadStatus({ type: 'idle', message: '' });

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:8000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok && data.status === 'success') {
        setUploadStatus({
          type: 'success',
          message: `Uploaded & extracted ${data.total_extracted_frames} frames @ ${data.fps} FPS into ${data.frames_dir}!`,
        });
      } else {
        setUploadStatus({
          type: 'error',
          message: data.detail || 'Upload failed. Ensure FastAPI backend is running.',
        });
      }
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : 'Upload failed';
      setUploadStatus({
        type: 'error',
        message: `Connection error: ${errorMsg}. Make sure backend is running at http://127.0.0.1:8000`,
      });
    } finally {
      setIsUploading(false);
    }
  };

  useEffect(() => {
    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [videoUrl]);

  return (
    <div className="app-container">
      <Header />

      <main className="main-grid">
        <VideoUploader
          onVideoSelect={handleVideoSelect}
          selectedFile={selectedFile}
          onUploadToBackend={handleUploadToBackend}
          isUploading={isUploading}
          uploadStatus={uploadStatus}
        />
        <VideoPreview
          videoUrl={videoUrl}
          videoFile={selectedFile}
          onClearVideo={handleClearVideo}
        />
      </main>

      <section style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <SearchBar />
        <ResultsSection />
      </section>

      <footer className="footer">
        Semantic Video Search Application &bull; Frame Extraction Prototype
      </footer>
    </div>
  );
}
