import React from 'react';
import { Layers, SearchX, Clock, Zap, Play, CheckCircle2, Loader2 } from 'lucide-react';

export interface SearchResult {
  similarity_score: number;
  similarity_percent: number;
  timestamp: number;
  formatted_timestamp: string;
  frame_image: string;
  frame_index: number;
  filename: string;
  video_id: string;
  caption?: string;
}

interface ResultsSectionProps {
  results: SearchResult[];
  activeQuery: string;
  selectedTimestamp: number | null;
  onSelectTimestamp: (timestamp: number) => void;
  isSearching: boolean;
}

export const ResultsSection: React.FC<ResultsSectionProps> = ({
  results,
  activeQuery,
  selectedTimestamp,
  onSelectTimestamp,
  isSearching,
}) => {
  return (
    <div className="card results-card">
      <div className="card-header">
        <div className="card-title">
          <Layers size={18} />
          <span>Top Matches (Above Threshold)</span>
        </div>
        {results.length > 0 && !isSearching && (
          <span style={{ fontSize: '0.8125rem', color: '#94a3b8' }}>
            Displaying Top {results.length} frame matches for &quot;{activeQuery}&quot; (Threshold &ge; 0.25)
          </span>
        )}
      </div>

      {isSearching ? (
        <div className="empty-results">
          <div className="empty-results-icon" style={{ borderColor: 'var(--primary)', color: 'var(--primary)' }}>
            <Loader2 size={32} className="animate-spin" />
          </div>
          <h3 className="empty-results-title">Computing Similarities & Vision Captions...</h3>
          <p className="empty-results-desc">
            Matching text vector embedding against local frame embeddings & reranking with AI captions.
          </p>
        </div>
      ) : results.length > 0 ? (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
            gap: '1.25rem',
            marginTop: '0.5rem'
          }}
        >
          {results.map((item, idx) => {
            const isSelected = selectedTimestamp === item.timestamp;

            return (
              <div
                key={`${item.video_id}-${item.frame_index}-${idx}`}
                onClick={() => onSelectTimestamp(item.timestamp)}
                className={`result-card ${isSelected ? 'active-card' : ''}`}
                style={{
                  background: '#0e131f',
                  border: isSelected ? '2px solid #6366f1' : '1px solid rgba(255, 255, 255, 0.08)',
                  boxShadow: isSelected ? '0 0 18px rgba(99, 102, 241, 0.45)' : 'none',
                  borderRadius: '12px',
                  overflow: 'hidden',
                  cursor: 'pointer',
                  transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                  display: 'flex',
                  flexDirection: 'column',
                  position: 'relative'
                }}
              >
                {/* Frame Image */}
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
                    src={item.frame_image}
                    alt={`Frame at ${item.formatted_timestamp}`}
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover'
                    }}
                    onError={(e) => {
                      (e.currentTarget as HTMLImageElement).src =
                        'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="100%" height="100%" fill="%231e293b"/></svg>';
                    }}
                  />

                  <div
                    style={{
                      position: 'absolute',
                      inset: 0,
                      background: 'rgba(0,0,0,0.3)',
                      opacity: isSelected ? 0.1 : 0,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      transition: 'opacity 0.2s ease'
                    }}
                    className="hover-play-overlay"
                  >
                    <div
                      style={{
                        width: '36px',
                        height: '36px',
                        borderRadius: '50%',
                        background: 'rgba(99, 102, 241, 0.9)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: '#fff'
                      }}
                    >
                      <Play size={18} style={{ marginLeft: '2px' }} />
                    </div>
                  </div>

                  {/* Timestamp Badge */}
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

                  {isSelected && (
                    <div
                      style={{
                        position: 'absolute',
                        top: '8px',
                        left: '8px',
                        padding: '0.2rem 0.45rem',
                        borderRadius: '6px',
                        background: '#6366f1',
                        fontSize: '0.7rem',
                        fontWeight: 600,
                        color: '#ffffff',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.25rem',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.4)'
                      }}
                    >
                      <CheckCircle2 size={12} />
                      Playing
                    </div>
                  )}
                </div>

                {/* Card Info */}
                <div style={{ padding: '0.875rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.8125rem', fontWeight: 500, color: '#f8fafc' }}>
                      <Clock size={13} style={{ color: '#38bdf8' }} />
                      <span>{item.formatted_timestamp} ({item.timestamp}s)</span>
                    </div>

                    <div
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.25rem',
                        padding: '0.2rem 0.5rem',
                        borderRadius: '9999px',
                        fontSize: '0.75rem',
                        fontWeight: 600,
                        background: isSelected ? 'rgba(99, 102, 241, 0.25)' : 'rgba(99, 102, 241, 0.12)',
                        color: isSelected ? '#ffffff' : '#a5b4fc',
                        border: `1px solid ${isSelected ? '#6366f1' : 'rgba(99, 102, 241, 0.3)'}`
                      }}
                    >
                      <Zap size={12} />
                      Score: {item.similarity_score}
                    </div>
                  </div>

                  {/* AI Frame Caption */}
                  {item.caption && (
                    <div
                      style={{
                        fontSize: '0.75rem',
                        color: '#94a3b8',
                        background: 'rgba(15, 23, 42, 0.6)',
                        padding: '0.35rem 0.5rem',
                        borderRadius: '6px',
                        border: '1px solid rgba(255, 255, 255, 0.05)',
                        lineHeight: 1.35,
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden'
                      }}
                      title={item.caption}
                    >
                      <strong style={{ color: '#818cf8', fontWeight: 600 }}>Caption: </strong>
                      {item.caption}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="empty-results">
          <div className="empty-results-icon">
            <SearchX size={28} />
          </div>
          <h3 className="empty-results-title">
            {activeQuery ? 'No Matching Frames Found' : 'No Search Results'}
          </h3>
          <p className="empty-results-desc">
            {activeQuery
              ? `No frames matched "${activeQuery}" above threshold (≥ 0.25). Try natural language examples: "person wearing red shirt", "man with black cap", "person holding a phone", "woman carrying a black handbag", "person wearing white shoes".`
              : 'Enter a natural language search query above (e.g., "person wearing red shirt", "man with black cap") to display matching frames.'}
          </p>
        </div>
      )}
    </div>
  );
};
