import React, { useEffect, useState } from 'react';
import { Video, Server } from 'lucide-react';

export const Header: React.FC = () => {
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/health');
        if (response.ok) {
          setBackendStatus('online');
        } else {
          setBackendStatus('offline');
        }
      } catch {
        setBackendStatus('offline');
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="header">
      <div className="header-brand">
        <div className="header-icon">
          <Video size={22} />
        </div>
        <div>
          <h1 className="header-title">Semantic Video Search</h1>
          <p className="header-subtitle">AI-Powered Video Retrieval Prototype</p>
        </div>
      </div>

      <div className="header-actions">
        <div className="status-badge">
          <span className="status-dot"></span>
          Prototype Mode
        </div>

        <div className={`status-badge ${backendStatus === 'online' ? 'online' : backendStatus === 'offline' ? 'offline' : ''}`}>
          <Server size={13} />
          {backendStatus === 'online' ? 'API Connected' : backendStatus === 'offline' ? 'API Offline' : 'Checking API...'}
        </div>
      </div>
    </header>
  );
};
