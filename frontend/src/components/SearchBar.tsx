import React, { useState } from 'react';
import { Search, Loader2, X, Sparkles } from 'lucide-react';

interface SearchBarProps {
  onSearch: (query: str) => void;
  isSearching: boolean;
  activeQuery: string;
}

export const SearchBar: React.FC<SearchBarProps> = ({ onSearch, isSearching, activeQuery }) => {
  const [query, setQuery] = useState(activeQuery);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  const handleClear = () => {
    setQuery('');
  };

  return (
    <div className="card search-card">
      <div className="card-header">
        <div className="card-title">
          <Sparkles size={18} style={{ color: '#818cf8' }} />
          <span>Semantic OpenCLIP Video Search</span>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="search-box-wrapper" style={{ display: 'flex', gap: '0.75rem' }}>
        <div style={{ position: 'relative', flex: 1, display: 'flex', alignItems: 'center' }}>
          <Search className="search-icon" size={18} />
          <input
            type="text"
            className="search-input"
            style={{ opacity: 1, cursor: 'text', paddingRight: '2.5rem' }}
            placeholder="Type anything (e.g., 'a red car turning left', 'a person walking in rain')..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isSearching}
            aria-label="Semantic Search Query"
          />
          {query && (
            <button
              type="button"
              onClick={handleClear}
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
    </div>
  );
};
