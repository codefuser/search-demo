import React, { useState } from 'react';
import { Search, Loader2, X, Sparkles, History, Trash2 } from 'lucide-react';

interface SearchBarProps {
  onSearch: (query: string) => void;
  isSearching: boolean;
  activeQuery: string;
  searchHistory: string[];
  onClearHistory: () => void;
}

export const SearchBar: React.FC<SearchBarProps> = ({
  onSearch,
  isSearching,
  activeQuery,
  searchHistory,
  onClearHistory,
}) => {
  const [query, setQuery] = useState(activeQuery);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && query.trim()) {
      e.preventDefault();
      onSearch(query.trim());
    }
  };

  const handleClearInput = () => {
    setQuery('');
  };

  const handleChipClick = (term: string) => {
    setQuery(term);
    onSearch(term);
  };

  return (
    <div className="card search-card">
      <div className="card-header">
        <div className="card-title">
          <Sparkles size={18} style={{ color: '#818cf8' }} />
          <span>Semantic OpenCLIP Search</span>
        </div>
        {searchHistory.length > 0 && (
          <button
            type="button"
            onClick={onClearHistory}
            className="btn-icon-danger"
            title="Clear Search History"
            style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.75rem' }}
          >
            <Trash2 size={13} />
            <span>Clear History</span>
          </button>
        )}
      </div>

      <form onSubmit={handleSubmit} className="search-box-wrapper" style={{ display: 'flex', gap: '0.75rem' }}>
        <div style={{ position: 'relative', flex: 1, display: 'flex', alignItems: 'center' }}>
          <Search className="search-icon" size={18} />
          <input
            type="text"
            className="search-input"
            style={{ opacity: 1, cursor: 'text', paddingRight: '2.5rem' }}
            placeholder="Type query and press Enter (e.g., 'man with black cap', 'person wearing red shirt')..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isSearching}
            aria-label="Semantic Search Query"
          />
          {query && (
            <button
              type="button"
              onClick={handleClearInput}
              style={{
                position: 'absolute',
                right: '1rem',
                background: 'none',
                border: 'none',
                color: '#64748b',
                cursor: 'pointer'
              }}
            >
              <X size={16} />
            </button>
          )}
        </div>

        <button
          type="submit"
          className="btn-upload"
          disabled={isSearching || !query.trim()}
          style={{ minWidth: '120px', justifyContent: 'center' }}
        >
          {isSearching ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              Searching...
            </>
          ) : (
            <>
              <Search size={16} />
              Search
            </>
          )}
        </button>
      </form>

      {/* Suggested Natural Language Query Chips & Recent History */}
      <div className="history-wrap">
        <span style={{ fontSize: '0.75rem', color: '#64748b', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
          <Sparkles size={12} style={{ color: '#818cf8' }} /> Examples:
        </span>
        {[
          'person wearing red shirt',
          'man with black cap',
          'person holding a phone',
          'woman carrying a black handbag',
          'person wearing white shoes'
        ].map((term) => (
          <button
            key={term}
            type="button"
            className="history-chip"
            style={{ background: 'rgba(99, 102, 241, 0.1)', color: '#a5b4fc', border: '1px solid rgba(99, 102, 241, 0.25)' }}
            onClick={() => handleChipClick(term)}
          >
            {term}
          </button>
        ))}
      </div>

      {/* Search History Chips */}
      {searchHistory.length > 0 && (
        <div className="history-wrap">
          <span style={{ fontSize: '0.75rem', color: '#64748b', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <History size={12} /> Recent:
          </span>
          {searchHistory.map((term, index) => (
            <button
              key={`${term}-${index}`}
              type="button"
              className="history-chip"
              onClick={() => handleChipClick(term)}
            >
              {term}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};
