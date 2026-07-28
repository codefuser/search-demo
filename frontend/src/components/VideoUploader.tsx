import React, { useRef, useState } from 'react';
import { UploadCloud, Film, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

interface VideoUploaderProps {
  onVideoSelect: (file: File) => void;
  selectedFile: File | null;
  onUploadToBackend: () => void;
  isUploading: boolean;
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
      alert('Invalid file format. Please select an MP4, MOV, or AVI video file.');
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
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
            {selectedFile ? selectedFile.name : 'Select or Drag & Drop Video'}
          </p>
          <p className="dropzone-desc">
            Supported Formats: <strong>.mp4, .mov, .avi</strong>
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', justifyContent: 'center' }}>
          <button
            type="button"
            className="btn-upload"
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
          >
            <UploadCloud size={16} />
            {selectedFile ? 'Change Video' : 'Browse Local Video'}
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
                  Uploading...
                </>
              ) : (
                <>
                  <UploadCloud size={16} />
                  Upload to FastAPI
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {uploadStatus.type !== 'idle' && (
        <div
          className={`status-banner ${uploadStatus.type}`}
          style={{
            padding: '0.75rem 1rem',
            borderRadius: '8px',
            fontSize: '0.85rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            background: uploadStatus.type === 'success' ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.12)',
            color: uploadStatus.type === 'success' ? '#34d399' : '#f87171',
            border: `1px solid ${uploadStatus.type === 'success' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`
          }}
        >
          {uploadStatus.type === 'success' ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
          <span>{uploadStatus.message}</span>
        </div>
      )}
    </div>
  );
};
