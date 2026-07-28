import React from 'react';
import { Search, AlertCircle } from 'lucide-react';

export const SearchBar: React.FC = () => {
  return (
    <div className="card search-card">
      <div className="card-header">
        <div className="card-title">
          <Search size={18} />
          <span>Semantic Search Query</span>
        </div>
      </div>

      <div className="search-box-wrapper">
        <Search className="search-icon" size={18} />
        <input
          type="text"
          className="search-input"
          placeholder="e.g. 'a red car turning left at night' (Disabled in Prototype)"
          disabled
          aria-label="Semantic Search Query (Disabled)"
        />
      </div>

      <div className="search-disabled-banner">
        <AlertCircle size={16} style={{ flexShrink: 0 }} />
        <span>
          Search input is disabled in this prototype layout. AI vector embedding & indexing will be attached in future releases.
        </span>
      </div>
    </div>
  );
};
