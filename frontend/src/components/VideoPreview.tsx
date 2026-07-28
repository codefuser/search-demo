import React, { useState } from 'react';
import { PlayCircle, Trash2, Video as VideoIcon, Clock, HardDrive, FileVideo } from 'lucide-react';

interface VideoPreviewProps {
  videoUrl: string | null;
  videoFile: File | null;
  onClearVideo: () => void;
}

export const VideoPreview: React.FC<VideoPreviewProps> = ({ videoUrl, videoFile, onClearVideo }) => {
  const [durationSec, setDurationSec] = useState<number | null>(null);

  const handleLoadedMetadata = (e: React.SyntheticEvent<HTMLVideoElement, Event>) => {
    const videoElem = e.currentTarget;
    if (videoElem.duration && !isNaN(videoElem.duration)) {
      setDurationSec(videoElem.duration);
    }
  };

  const formatDuration = (seconds: number | null): string => {
    if (seconds === null || isNaN(seconds)) return 'Calculating...';
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    const pad = (n: number) => n.toString().padStart(2, '0');

    if (hrs > 0) {
      return `${pad(hrs)}:${pad(mins)}:${pad(secs)}`;
    }
    return `${pad(mins)}:${pad(secs)}`;
  };

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
          <span>Video Preview & Details</span>
        </div>
        {videoFile && (
          <button
            type="button"
            className="btn-icon-danger"
            onClick={() => {
              setDurationSec(null);
              onClearVideo();
            }}
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
            onLoadedMetadata={handleLoadedMetadata}
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
                Select a local video file (.mp4, .mov, .avi) to view preview and metadata
              </p>
            </div>
          </div>
        )}
      </div>

      {videoFile && (
        <div className="video-metadata-panel" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <div className="video-info-bar">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', overflow: 'hidden' }}>
              <FileVideo size={16} style={{ flexShrink: 0, color: '#818cf8' }} />
              <span className="video-name" title={videoFile.name}>
                {videoFile.name}
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', flexShrink: 0 }}>
              <HardDrive size={14} style={{ color: '#94a3b8' }} />
              <span>{formatFileSize(videoFile.size)}</span>
            </div>
          </div>

          <div className="video-info-bar" style={{ background: 'rgba(99, 102, 241, 0.08)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Clock size={16} style={{ color: '#38bdf8' }} />
              <span style={{ fontWeight: 500, color: '#f8fafc' }}>Duration:</span>
            </div>
            <span style={{ fontWeight: 600, color: '#38bdf8', fontFamily: 'monospace' }}>
              {formatDuration(durationSec)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
