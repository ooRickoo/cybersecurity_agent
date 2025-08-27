import React from 'react';
import { useQuery } from 'react-query';
import { Activity, HardDrive, Clock, Wifi } from 'lucide-react';
import axios from 'axios';

const StatusBar = ({ isConnected }) => {
  // Fetch system status
  const { data: statusData } = useQuery(
    'status',
    async () => {
      const response = await axios.get('/api/status');
      return response.data;
    },
    {
      refetchInterval: 10000, // Refresh every 10 seconds
    }
  );

  const status = statusData?.status || {};

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'Unknown';
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <footer className="bg-cyber-light/80 backdrop-blur-sm border-t border-cyber-accent/20 px-4 py-2">
      <div className="flex items-center justify-between text-sm">
        {/* Left Section */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}></div>
            <span className="text-cyber-accent/70">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          
          <div className="hidden md:flex items-center gap-2 text-cyber-accent/70">
            <Activity size={14} />
            <span>Server: {status.server || 'Unknown'}</span>
          </div>
        </div>

        {/* Center Section */}
        <div className="hidden lg:flex items-center gap-6 text-cyber-accent/70">
          <div className="flex items-center gap-2">
            <HardDrive size={14} />
            <span>{status.sessionsCount || 0} Sessions</span>
          </div>
          
          <div className="flex items-center gap-2">
            <Wifi size={14} />
            <span>{status.totalFiles || 0} Files</span>
          </div>
          
          <div className="flex items-center gap-2">
            <Clock size={14} />
            <span>Updated: {formatTimestamp(status.timestamp)}</span>
          </div>
        </div>

        {/* Right Section */}
        <div className="flex items-center gap-4 text-cyber-accent/70">
          <div className="hidden sm:flex items-center gap-2">
            <span>Port: 3001</span>
          </div>
          
          <div className="text-xs">
            Cybersecurity Agent Agent v1.0
          </div>
        </div>
      </div>
    </footer>
  );
};

export default StatusBar;
