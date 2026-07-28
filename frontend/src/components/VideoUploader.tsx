import React, { useRef, useState } from 'react';
import { UploadCloud, Film } from 'lucide-react';

interface VideoUploaderProps {
  onVideoSelect: (file: File) => void;
  selectedFile: File | null;
}

export const VideoUploader: React.FC<VideoUploaderProps> = ({ onVideoSelect, selectedFile }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onVideoSelect(e.target.files[0]);
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
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('video/')) {
        onVideoSelect(file);
      }
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <Film size={18} className="text-primary" />
          <span>Upload Video</span>
        </div>
      </div>

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="video/*"
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
            Supports MP4, WEBM, MKV or MOV video files
          </p>
        </div>
        <button type="button" className="btn-upload" onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}>
          <UploadCloud size={16} />
          {selectedFile ? 'Change Video' : 'Upload Video'}
        </button>
      </div>
    </div>
  );
};
