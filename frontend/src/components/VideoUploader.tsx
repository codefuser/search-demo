import React, { useRef, useState } from 'react';
import { UploadCloud, Film, CheckCircle, AlertTriangle, Loader2, RefreshCw } from 'lucide-react';

interface VideoUploaderProps {
  onVideoSelect: (file: File) => void;
  selectedFile: File | null;
  onUploadToBackend: () => void;
  isUploading: boolean;
  indexingProgress: number;
  uploadStatus: {
    type: 'idle' | 'success' | 'error';
    message: string;
  };
}

export const VideoUploader: React.FC<VideoUploaderProps> = ({
  onVideoSelect,
  selectedFile,
  onUploadToBackend,
  isUploading,
  indexingProgress,
  uploadStatus,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSelect(e.target.files[0]);
    }
  };

  const validateAndSelect = (file: File) => {
    const validExtensions = ['.mp4', '.mov', '.avi'];
    const fileNameLower = file.name.toLowerCase();
    const isValid = validExtensions.some((ext) => fileNameLower.endsWith(ext));

    if (isValid) {
      onVideoSelect(file);
    } else {
      alert(`Invalid format "${file.name}". Please select a valid video file (.mp4, .mov, .avi).`);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSelect(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <Film size={18} className="text-primary" />
          <span>Upload Local Video</span>
        </div>
      </div>

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="video/mp4,video/quicktime,video/x-msvideo,.mp4,.mov,.avi"
        style={{ display: 'none' }}
        id="video-upload-input"
      />

      <div
        className={`dropzone ${isDragging ? 'active' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="dropzone-icon-wrap">
          <UploadCloud size={26} />
        </div>
        <div>
          <p className="dropzone-title">
            {selectedFile ? selectedFile.name : 'Drag & Drop Local Video Here'}
          </p>
          <p className="dropzone-desc">
            Supports <strong>.mp4, .mov, .avi</strong> video formats
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', justifyContent: 'center' }}>
          <button
            type="button"
            className="btn-upload"
            disabled={isUploading}
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
          >
            <UploadCloud size={16} />
            {selectedFile ? 'Change File' : 'Browse Files'}
          </button>

          {selectedFile && (
            <button
              type="button"
              className="btn-upload"
              style={{ background: '#10b981' }}
              disabled={isUploading}
              onClick={(e) => {
                e.stopPropagation();
                onUploadToBackend();
              }}
            >
              {isUploading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Indexing ({indexingProgress}%)
                </>
              ) : (
                <>
                  <UploadCloud size={16} />
                  Upload & Extract Index
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar while Indexing */}
      {isUploading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#94a3b8' }}>
            <span>Extracting frames & indexing OpenCLIP vectors...</span>
            <span style={{ fontWeight: 600, color: '#6366f1' }}>{indexingProgress}%</span>
          </div>
          <div className="progress-bar-wrap">
            <div className="progress-bar-fill" style={{ width: `${indexingProgress}%` }}></div>
          </div>
        </div>
      )}

      {/* Status & Error Handling Banner */}
      {uploadStatus.type !== 'idle' && (
        <div
          className={`status-banner ${uploadStatus.type}`}
          style={{
            padding: '0.75rem 1rem',
            borderRadius: '8px',
            fontSize: '0.85rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '0.5rem',
            background: uploadStatus.type === 'success' ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.12)',
            color: uploadStatus.type === 'success' ? '#34d399' : '#f87171',
            border: `1px solid ${uploadStatus.type === 'success' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {uploadStatus.type === 'success' ? <CheckCircle size={16} /> : <AlertTriangle size={16} />}
            <span>{uploadStatus.message}</span>
          </div>

          {uploadStatus.type === 'error' && (
            <button
              type="button"
              onClick={onUploadToBackend}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#f87171',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem',
                fontSize: '0.75rem',
                fontWeight: 600
              }}
            >
              <RefreshCw size={12} /> Retry
            </button>
          )}
        </div>
      )}
    </div>
  );
};
