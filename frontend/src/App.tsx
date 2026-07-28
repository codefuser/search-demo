import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { VideoUploader } from './components/VideoUploader';
import { VideoPreview } from './components/VideoPreview';
import { SearchBar } from './components/SearchBar';
import { ResultsSection, SearchResult } from './components/ResultsSection';

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [currentVideoId, setCurrentVideoId] = useState<string | null>(null);

  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadStatus, setUploadStatus] = useState<{
    type: 'idle' | 'success' | 'error';
    message: string;
  }>({ type: 'idle', message: '' });

  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [activeQuery, setActiveQuery] = useState<string>('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [seekTimestamp, setSeekTimestamp] = useState<number | null>(null);

  const handleVideoSelect = (file: File) => {
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setCurrentVideoId(null);
    setSearchResults([]);
    setActiveQuery('');
    setSeekTimestamp(null);
    setUploadStatus({ type: 'idle', message: '' });
  };

  const handleClearVideo = () => {
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }
    setSelectedFile(null);
    setVideoUrl(null);
    setCurrentVideoId(null);
    setSearchResults([]);
    setActiveQuery('');
    setSeekTimestamp(null);
    setUploadStatus({ type: 'idle', message: '' });
  };

  const handleUploadToBackend = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setUploadStatus({
      type: 'idle',
      message: 'Extracting frames & indexing OpenCLIP vector embeddings...',
    });

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:8000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok && data.status === 'success') {
        setCurrentVideoId(data.video_id);
        const cacheTag = data.is_cached ? '⚡ Loaded instantly from RAM cache' : 'Indexed & cached';
        setUploadStatus({
          type: 'success',
          message: `Indexing complete (${cacheTag}): ${data.indexed_embeddings_count} frame vectors ready for fast search!`,
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

  const handleSearch = async (queryText: string) => {
    if (!queryText.trim()) return;

    setIsSearching(true);
    setActiveQuery(queryText);

    try {
      const response = await fetch('http://127.0.0.1:8000/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: queryText,
          video_id: currentVideoId || undefined,
          top_k: 6,
        }),
      });

      const data = await response.json();

      if (response.ok && data.status === 'success') {
        setSearchResults(data.results || []);
        if (data.results && data.results.length > 0) {
          setSeekTimestamp(data.results[0].timestamp);
        }
      } else {
        setSearchResults([]);
        alert(`Search error: ${data.detail || 'Failed to perform semantic search'}`);
      }
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : 'Search request failed';
      setSearchResults([]);
      alert(`Search failed: ${errorMsg}. Ensure FastAPI backend is running.`);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSelectTimestamp = (timestamp: number) => {
    setSeekTimestamp(timestamp);
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
          seekTimestamp={seekTimestamp}
          onClearVideo={handleClearVideo}
        />
      </main>

      <section style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <SearchBar
          onSearch={handleSearch}
          isSearching={isSearching}
          activeQuery={activeQuery}
        />
        <ResultsSection
          results={searchResults}
          activeQuery={activeQuery}
          selectedTimestamp={seekTimestamp}
          onSelectTimestamp={handleSelectTimestamp}
          isSearching={isSearching}
        />
      </section>

      <footer className="footer">
        Semantic Video Search Application &bull; Cached In-Memory Vector Search
      </footer>
    </div>
  );
}
