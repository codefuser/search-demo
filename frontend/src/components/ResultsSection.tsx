import React from 'react';
import { Layers, SearchX } from 'lucide-react';

export const ResultsSection: React.FC = () => {
  return (
    <div className="card results-card">
      <div className="card-header">
        <div className="card-title">
          <Layers size={18} />
          <span>Search Results</span>
        </div>
      </div>

      <div className="empty-results">
        <div className="empty-results-icon">
          <SearchX size={28} />
        </div>
        <h3 className="empty-results-title">No Search Results Yet</h3>
        <p className="empty-results-desc">
          When video indexing and semantic AI query capabilities are integrated, matching video timestamps and segment clips will appear here.
        </p>
      </div>
    </div>
  );
};
