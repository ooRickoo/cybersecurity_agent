import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from 'react-query';
import { 
  FolderOpen, 
  FileText, 
  Image, 
  Database, 
  BarChart3, 
  Clock, 
  HardDrive,
  Activity,
  TrendingUp,
  Shield,
  Zap,
  Eye
} from 'lucide-react';
import { motion } from 'framer-motion';
import axios from 'axios';

const Dashboard = ({ onRouteChange }) => {
  const navigate = useNavigate();
  // const [selectedSession, setSelectedSession] = useState(null); // Commented out until needed

  // Fetch sessions data
  const { data: sessionsData, isLoading, error, refetch } = useQuery(
    'sessions',
    async () => {
      const response = await axios.get('/api/sessions');
      return response.data;
    },
    {
      refetchInterval: 10000, // Refresh every 10 seconds
      staleTime: 30000,
    }
  );

  // Fetch system status
  const { data: statusData } = useQuery(
    'status',
    async () => {
      const response = await axios.get('/api/status');
      return response.data;
    },
    {
      refetchInterval: 5000, // Refresh every 5 seconds
    }
  );

  const sessions = sessionsData?.sessions || [];
  const status = statusData?.status || {};

  // Calculate statistics
  const stats = {
    totalSessions: sessions.length,
    totalFiles: sessions.reduce((sum, session) => sum + session.fileCount, 0),
    totalSize: sessions.reduce((sum, session) => sum + session.totalSize, 0),
    recentSessions: sessions.slice(0, 5),
    fileTypes: getFileTypeDistribution(sessions),
  };

  function getFileTypeDistribution(sessions) {
    const distribution = {};
    sessions.forEach(session => {
      session.files?.forEach(file => {
        const type = file.type || 'unknown';
        distribution[type] = (distribution[type] || 0) + 1;
      });
    });
    return distribution;
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileTypeIcon = (type) => {
    switch (type) {
      case 'image': return <Image size={16} />;
      case 'document': return <FileText size={16} />;
      case 'data': return <BarChart3 size={16} />;
      case 'database': return <Database size={16} />;
      case 'archive': return <FolderOpen size={16} />;
      default: return <FileText size={16} />;
    }
  };

  const getFileTypeColor = (type) => {
    switch (type) {
      case 'image': return 'from-blue-500 to-purple-600';
      case 'document': return 'from-green-500 to-teal-600';
      case 'data': return 'from-orange-500 to-red-600';
      case 'database': return 'from-indigo-500 to-blue-600';
      case 'archive': return 'from-purple-500 to-pink-600';
      default: return 'from-gray-500 to-gray-600';
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="cyber-spinner"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="text-accent-500 text-6xl mb-4">⚠️</div>
          <h2 className="text-xl font-semibold text-white mb-2">Error Loading Dashboard</h2>
          <p className="text-cyber-accent/70 mb-4">{error.message}</p>
          <button 
            onClick={() => refetch()}
            className="cyber-button"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">
              {/* Welcome Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="cyber-panel bg-gradient-to-br from-cyber-dark/80 via-cyber-light/60 to-cyber-dark/80"
        >
          <div className="cyber-panel-header">
            <div className="cyber-panel-title">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyber-accent to-cyber-accent3 flex items-center justify-center">
                <Shield size={28} className="text-cyber-dark" />
              </div>
              <div>
                <h1 className="text-3xl font-bold cyber-gradient-text">
                  Cybersecurity Agent Agent
                </h1>
                <p className="text-lg text-cyber-accent/80 font-medium">
                  Session Viewer & Workflow Output Manager
                </p>
              </div>
            </div>
            <div className="cyber-panel-subtitle">
              Professional interface for monitoring and analyzing cybersecurity workflow outputs in real-time
            </div>
          </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="cyber-card cyber-card-hover p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyber-accent to-cyber-accent3 flex items-center justify-center">
                <FolderOpen size={20} className="text-cyber-dark" />
              </div>
              <div>
                <div className="text-2xl font-bold text-white">{stats.totalSessions}</div>
                <div className="text-sm text-cyber-accent/70">Active Sessions</div>
              </div>
            </div>
          </div>

          <div className="cyber-card cyber-card-hover p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-green-500 to-teal-600 flex items-center justify-center">
                <FileText size={20} className="text-white" />
              </div>
              <div>
                <div className="text-2xl font-bold text-white">{stats.totalFiles}</div>
                <div className="text-sm text-cyber-accent/70">Total Files</div>
              </div>
            </div>
          </div>

          <div className="cyber-card cyber-card-hover p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center">
                <HardDrive size={20} className="text-white" />
              </div>
              <div>
                <div className="text-2xl font-bold text-white">{formatFileSize(stats.totalSize)}</div>
                <div className="text-sm text-cyber-accent/70">Total Size</div>
              </div>
            </div>
          </div>

          <div className="cyber-card cyber-card-hover p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center">
                <Activity size={20} className="text-white" />
              </div>
              <div>
                <div className="text-2xl font-bold text-white">{status.server === 'running' ? 'Live' : 'Offline'}</div>
                <div className="text-sm text-cyber-accent/70">System Status</div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Recent Sessions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="cyber-panel"
      >
        <div className="cyber-panel-header">
          <div className="cyber-panel-title">
            <Clock className="text-cyber-accent" />
            Recent Sessions
          </div>
          <div className="cyber-panel-subtitle">
            Latest workflow outputs and analysis results
          </div>
        </div>

        {stats.recentSessions.length > 0 ? (
          <div className="cyber-list">
            {stats.recentSessions.map((session, index) => (
              <motion.div
                key={session.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="cyber-list-item cursor-pointer"
                onClick={() => navigate(`/sessions/${session.id}`)}
              >
                <div className="cyber-list-item-icon">
                  <FolderOpen size={16} />
                </div>
                <div className="cyber-list-item-content">
                  <div className="cyber-list-item-title">{session.name}</div>
                  <div className="cyber-list-item-subtitle">
                    {session.fileCount} files • {session.totalSizeFormatted}
                  </div>
                </div>
                <div className="cyber-list-item-meta">
                  {new Date(session.lastModified).toLocaleDateString()}
                </div>
                <button className="cyber-button-secondary text-xs">
                  <Eye size={14} className="mr-1" />
                  View
                </button>
              </motion.div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <div className="text-cyber-accent/50 text-6xl mb-4">📁</div>
            <h3 className="text-lg font-semibold text-white mb-2">No Sessions Found</h3>
            <p className="text-cyber-accent/70">Start a cybersecurity workflow to see outputs here</p>
          </div>
        )}
      </motion.div>

      {/* File Type Distribution */}
      {Object.keys(stats.fileTypes).length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="cyber-panel"
        >
          <div className="cyber-panel-header">
            <div className="cyber-panel-title">
              <BarChart3 className="text-cyber-accent" />
              File Type Distribution
            </div>
            <div className="cyber-panel-subtitle">
              Overview of output file types across all sessions
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {Object.entries(stats.fileTypes).map(([type, count]) => (
              <div key={type} className="cyber-card cyber-card-hover p-4 text-center">
                <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${getFileTypeColor(type)} flex items-center justify-center mx-auto mb-3`}>
                  {getFileTypeIcon(type)}
                </div>
                <div className="text-lg font-bold text-white">{count}</div>
                <div className="text-xs text-cyber-accent/70 capitalize">{type}</div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Quick Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="cyber-panel"
      >
        <div className="cyber-panel-header">
          <div className="cyber-panel-title">
            <Zap className="text-cyber-accent" />
            Quick Actions
          </div>
          <div className="cyber-panel-subtitle">
            Common tasks and shortcuts
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button 
            onClick={() => navigate('/sessions')}
            className="cyber-card cyber-card-hover p-6 text-center hover:scale-105 transition-transform duration-200"
          >
            <div className="w-16 h-16 rounded-lg bg-gradient-to-br from-cyber-accent to-cyber-accent3 flex items-center justify-center mx-auto mb-4">
              <FolderOpen size={32} className="text-cyber-dark" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">Browse Sessions</h3>
            <p className="text-sm text-cyber-accent/70">View all available workflow sessions</p>
          </button>

          <button 
            onClick={() => refetch()}
            className="cyber-card cyber-card-hover p-6 text-center hover:scale-105 transition-transform duration-200"
          >
            <div className="w-16 h-16 rounded-lg bg-gradient-to-br from-green-500 to-teal-600 flex items-center justify-center mx-auto mb-4">
              <Activity size={32} className="text-white" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">Refresh Data</h3>
            <p className="text-sm text-cyber-accent/70">Update session information</p>
          </button>

          <button 
            className="cyber-card cyber-card-hover p-6 text-center hover:scale-105 transition-transform duration-200"
          >
            <div className="w-16 h-16 rounded-lg bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center mx-auto mb-4">
              <TrendingUp size={32} className="text-white" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">Analytics</h3>
            <p className="text-sm text-cyber-accent/70">View detailed session analytics</p>
          </button>
        </div>
      </motion.div>
    </div>
  );
};

export default Dashboard;
