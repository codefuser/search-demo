import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { VideoUploader } from './components/VideoUploader';
import { VideoPreview } from './components/VideoPreview';
import { SearchBar } from './components/SearchBar';
import { ResultsSection } from './components/ResultsSection';

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  const handleVideoSelect = (file: File) => {
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
  };

  const handleClearVideo = () => {
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }
    setSelectedFile(null);
    setVideoUrl(null);
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
        <VideoUploader onVideoSelect={handleVideoSelect} selectedFile={selectedFile} />
        <VideoPreview videoUrl={videoUrl} videoFile={selectedFile} onClearVideo={handleClearVideo} />
      </main>

      <section style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <SearchBar />
        <ResultsSection />
      </section>

      <footer className="footer">
        Semantic Video Search Application &bull; Prototype Folder Structure & Layout
      </footer>
    </div>
  );
}
