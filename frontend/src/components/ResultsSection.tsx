import React from 'react';
import { Layers, SearchX, Clock, Zap, Play } from 'lucide-react';

export interface SearchResult {
  video_id: string;
  frame_index: number;
  filename: string;
  frame_url: string;
  timestamp: number;
  formatted_timestamp: string;
  raw_score: number;
  similarity_percent: number;
}

interface ResultsSectionProps {
  results: SearchResult[];
  activeQuery: string;
  onSelectTimestamp: (timestamp: number) => void;
  isSearching: boolean;
}

export const ResultsSection: React.FC<ResultsSectionProps> = ({
  results,
  activeQuery,
  onSelectTimestamp,
  isSearching,
}) => {
  return (
    <div className="card results-card">
      <div className="card-header">
        <div className="card-title">
          <Layers size={18} />
          <span>Semantic Search Results</span>
        </div>
        {results.length > 0 && (
          <span style={{ fontSize: '0.8125rem', color: '#94a3b8' }}>
            Found {results.length} matching frame clips for &quot;{activeQuery}&quot;
          </span>
        )}
      </div>

      {results.length > 0 ? (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
            gap: '1.25rem',
            marginTop: '0.5rem'
          }}
        >
          {results.map((item, idx) => (
            <div
              key={`${item.video_id}-${item.frame_index}-${idx}`}
              onClick={() => onSelectTimestamp(item.timestamp)}
              className="result-card"
              style={{
                background: '#0e131f',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: '12px',
                overflow: 'hidden',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                display: 'flex',
                flexDirection: 'column'
              }}
            >
              <div
                style={{
                  position: 'relative',
                  width: '100%',
                  aspectRatio: '16 / 9',
                  background: '#000',
                  overflow: 'hidden'
                }}
              >
                <img
                  src={item.frame_url}
                  alt={`Frame ${item.frame_index}`}
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover'
                  }}
                  onError={(e) => {
                    // Fallback placeholder if image load fails
                    (e.currentTarget as HTMLImageElement).src =
                      'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="100%" height="100%" fill="%231e293b"/></svg>';
                  }}
                />
                
                <div
                  style={{
                    position: 'absolute',
                    inset: 0,
                    background: 'rgba(0,0,0,0.3)',
                    opacity: 0,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'opacity 0.2s ease'
                  }}
                  className="hover-play-overlay"
                >
                  <div
                    style={{
                      width: '40px',
                      height: '40px',
                      borderRadius: '50%',
                      background: 'rgba(99, 102, 241, 0.9)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#fff'
                    }}
                  >
                    <Play size={20} style={{ marginLeft: '2px' }} />
                  </div>
                </div>

                <div
                  style={{
                    position: 'absolute',
                    top: '8px',
                    right: '8px',
                    padding: '0.25rem 0.5rem',
                    borderRadius: '6px',
                    background: 'rgba(15, 23, 42, 0.85)',
                    backdropFilter: 'blur(4px)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    color: '#38bdf8',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.25rem'
                  }}
                >
                  <Clock size={12} />
                  {item.formatted_timestamp}
                </div>
              </div>

              <div style={{ padding: '0.875rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                    Frame #{item.frame_index}
                  </span>
                  <div
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.25rem',
                      padding: '0.2rem 0.5rem',
                      borderRadius: '9999px',
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      background: 'rgba(99, 102, 241, 0.15)',
                      color: '#a5b4fc',
                      border: '1px solid rgba(99, 102, 241, 0.3)'
                    }}
                  >
                    <Zap size={12} />
                    {item.similarity_percent}% Match
                  </div>
                </div>

                <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
                  Click card to play at {item.formatted_timestamp}
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="empty-results">
          <div className="empty-results-icon">
            <SearchX size={28} />
          </div>
          <h3 className="empty-results-title">
            {isSearching ? 'Searching Video Frames...' : activeQuery ? 'No Matching Frames Found' : 'No Search Query Entered'}
          </h3>
          <p className="empty-results-desc">
            {activeQuery
              ? `No frames closely matched "${activeQuery}". Try another descriptive phrase.`
              : 'Upload a video and type any natural language phrase (e.g. "a person running", "a car turning") to search frames.'}
          </p>
        </div>
      )}
    </div>
  );
};
