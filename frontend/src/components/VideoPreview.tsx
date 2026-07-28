import React from 'react';
import { PlayCircle, Trash2, Video as VideoIcon } from 'lucide-react';

interface VideoPreviewProps {
  videoUrl: string | null;
  videoFile: File | null;
  onClearVideo: () => void;
}

export const VideoPreview: React.FC<VideoPreviewProps> = ({ videoUrl, videoFile, onClearVideo }) => {
  const formatFileSize = (bytes: number) => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <PlayCircle size={18} />
          <span>Video Preview</span>
        </div>
        {videoFile && (
          <button
            type="button"
            className="btn-icon-danger"
            onClick={onClearVideo}
            title="Remove Video"
          >
            <Trash2 size={16} />
          </button>
        )}
      </div>

      <div className="preview-container">
        {videoUrl ? (
          <video
            controls
            src={videoUrl}
            className="preview-video"
            key={videoUrl}
          >
            Your browser does not support the video tag.
          </video>
        ) : (
          <div className="preview-empty">
            <VideoIcon size={44} strokeWidth={1.5} />
            <div>
              <p style={{ fontWeight: 500 }}>No Video Loaded</p>
              <p style={{ fontSize: '0.8125rem', marginTop: '0.25rem' }}>
                Upload a video to display preview here
              </p>
            </div>
          </div>
        )}
      </div>

      {videoFile && (
        <div className="video-info-bar">
          <span className="video-name">{videoFile.name}</span>
          <span>{formatFileSize(videoFile.size)}</span>
        </div>
      )}
    </div>
  );
};
