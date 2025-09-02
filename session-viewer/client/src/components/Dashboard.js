import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from 'react-query';
import { 
  FolderOpen, 
  FileText, 
  HardDrive,
  Activity,
  Shield,
  Zap,
  Search,
  ArrowRight,
  Workflow,
  Target,
  FileSpreadsheet,
  FileCode,
  FileImage,
  FileArchive,
  RotateCcw,
  Settings,
  Download,
  BarChart3
} from 'lucide-react';
import { motion } from 'framer-motion';
import axios from 'axios';

const Dashboard = ({ onRouteChange }) => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');

  // Fetch sessions data
  const { data: sessionsData, isLoading, error, refetch } = useQuery(
    'sessions',
    async () => {
      console.log('🔍 Fetching sessions data...');
      const response = await axios.get('/api/sessions');
      console.log('📊 Sessions API response:', response.data);
      return response.data;
    },
    {
      refetchInterval: 30000,
      staleTime: 60000,
    }
  );

  // Fetch system status
  const { data: statusData } = useQuery(
    'status',
    async () => {
      console.log('🔍 Fetching status data...');
      const response = await axios.get('/api/status');
      console.log('📊 Status API response:', response.data);
      return response.data;
    },
    {
      refetchInterval: 15000,
      staleTime: 30000,
    }
  );

  const sessions = sessionsData?.sessions || [];
  const status = statusData?.status || {};

  console.log('🔄 Dashboard render - sessions:', sessions);
  console.log('🔄 Dashboard render - status:', status);
  console.log('🔄 Dashboard render - isLoading:', isLoading);
  console.log('🔄 Dashboard render - error:', error);

  // Enhanced session analysis
  const analyzeSession = (session) => {
    const files = session.files || [];
    const hasCSV = files.some(f => f.name?.endsWith('.csv'));
    const hasPDF = files.some(f => f.name?.endsWith('.pdf'));
    const hasSummary = files.some(f => f.name?.includes('summary'));
    const hasWorkflow = files.some(f => f.name?.includes('workflow'));
    
    return {
      hasCSV,
      hasPDF,
      hasSummary,
      hasWorkflow,
      outputTypes: {
        data: files.filter(f => f.type === 'data').length,
        document: files.filter(f => f.type === 'document').length,
        image: files.filter(f => f.type === 'image').length,
        unknown: files.filter(f => f.type === 'unknown').length
      },
      mainOutputs: files.filter(f => 
        f.name?.endsWith('.csv') || 
        f.name?.includes('summary') || 
        f.name?.includes('report')
      ),
      resources: files.filter(f => 
        f.name?.endsWith('.pdf') || 
        f.name?.includes('resource')
      )
    };
  };

  // Filter sessions
  const filteredSessions = sessions.filter(session => {
    const matchesSearch = session.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         session.id?.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterType === 'all' || 
                         (filterType === 'with-outputs' && session.fileCount > 0) ||
                         (filterType === 'recent' && session.lastModified > Date.now() - 24*60*60*1000);
    return matchesSearch && matchesFilter;
  });

  // Calculate statistics
  const stats = {
    totalSessions: sessions.length,
    activeSessions: sessions.filter(s => s.fileCount > 0).length,
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

  const getFileIcon = (fileName) => {
    if (fileName?.endsWith('.csv')) return <FileSpreadsheet size={16} />;
    if (fileName?.endsWith('.pdf')) return <FileText size={16} />;
    if (fileName?.endsWith('.json')) return <FileCode size={16} />;
    if (fileName?.match(/\.(png|jpg|jpeg|svg)$/)) return <FileImage size={16} />;
    if (fileName?.match(/\.(zip|tar|gz)$/)) return <FileArchive size={16} />;
    return <FileText size={16} />;
  };

  const getWorkflowType = (session) => {
    const files = session.files || [];
    if (files.some(f => f.name?.includes('patent'))) return 'Patent Analysis';
    if (files.some(f => f.name?.includes('threat'))) return 'Threat Hunting';
    if (files.some(f => f.name?.includes('compliance'))) return 'Compliance';
    if (files.some(f => f.name?.includes('graph'))) return 'Graph Analysis';
    return 'General Analysis';
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
      {/* Executive Summary Header */}
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
                Cybersecurity Agent
              </h1>
              <p className="text-lg text-cyber-accent/80 font-medium">
                Executive Dashboard & Workflow Insights
              </p>
            </div>
          </div>
          <div className="cyber-panel-subtitle">
            Real-time overview of cybersecurity workflow outputs, key insights, and actionable results
          </div>
        </div>
      </motion.div>

      {/* Key Metrics Dashboard */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="cyber-card cyber-card-hover p-4"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyber-accent to-cyber-accent3 flex items-center justify-center">
              <FolderOpen size={20} className="text-cyber-dark" />
            </div>
            <div>
              <div className="text-2xl font-bold text-white">{stats.activeSessions}</div>
              <div className="text-sm text-cyber-accent/70">Active Sessions</div>
            </div>
          </div>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="cyber-card cyber-card-hover p-4"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-green-500 to-teal-600 flex items-center justify-center">
              <FileText size={20} className="text-white" />
            </div>
            <div>
              <div className="text-2xl font-bold text-white">{stats.totalFiles}</div>
              <div className="text-sm text-cyber-accent/70">Total Outputs</div>
            </div>
          </div>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="cyber-card cyber-card-hover p-4"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center">
              <HardDrive size={20} className="text-white" />
            </div>
            <div>
              <div className="text-2xl font-bold text-white">{formatFileSize(stats.totalSize)}</div>
              <div className="text-sm text-cyber-accent/70">Total Size</div>
            </div>
          </div>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="cyber-card cyber-card-hover p-4"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center">
              <Activity size={20} className="text-white" />
            </div>
            <div>
              <div className="text-2xl font-bold text-white">{status.server === 'running' ? 'Live' : 'Offline'}</div>
              <div className="text-sm text-cyber-accent/70">System Status</div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Search and Filter Bar */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="cyber-card p-4"
      >
        <div className="flex flex-col md:flex-row gap-4 items-center">
          <div className="flex-1 relative">
            <Search size={20} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-cyber-accent/50" />
            <input
              type="text"
              placeholder="Search sessions by name or ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-cyber-dark/50 border border-cyber-accent/30 rounded-lg text-white placeholder-cyber-accent/50 focus:border-cyber-accent focus:outline-none"
            />
          </div>
          <div className="flex gap-2">
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="px-4 py-2 bg-cyber-dark/50 border border-cyber-accent/30 rounded-lg text-white focus:border-cyber-accent focus:outline-none"
            >
              <option value="all">All Sessions</option>
              <option value="with-outputs">With Outputs</option>
              <option value="recent">Recent (24h)</option>
            </select>
                          <button
                onClick={() => refetch()}
                className="cyber-button-small"
              >
                <RotateCcw size={16} />
              </button>
          </div>
        </div>
      </motion.div>

      {/* Sessions Overview */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="space-y-4"
      >
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
          <Workflow size={24} className="text-cyber-accent" />
          Workflow Sessions
        </h2>
        
        {filteredSessions.length === 0 ? (
          <div className="cyber-card p-8 text-center">
            <div className="text-cyber-accent/50 text-6xl mb-4">📁</div>
            <h3 className="text-xl font-semibold text-white mb-2">No Sessions Found</h3>
            <p className="text-cyber-accent/70">Try adjusting your search or filter criteria</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {filteredSessions.map((session, index) => {
              const analysis = analyzeSession(session);
              const workflowType = getWorkflowType(session);
              
              return (
                <motion.div
                  key={session.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 * index }}
                  className="cyber-card cyber-card-hover p-6 cursor-pointer"
                  onClick={() => navigate(`/session/${session.id}`)}
                >
                  {/* Session Header */}
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyber-accent to-cyber-accent3 flex items-center justify-center">
                          <Target size={16} className="text-cyber-dark" />
                        </div>
                        <span className="text-sm font-medium text-cyber-accent/70">{workflowType}</span>
                      </div>
                      <h3 className="text-lg font-semibold text-white mb-1">{session.name}</h3>
                      <p className="text-sm text-cyber-accent/60 font-mono">{session.id}</p>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-cyber-accent/70">
                        {session.lastModified ? new Date(session.lastModified).toLocaleDateString() : 'Unknown'}
                      </div>
                      <div className="text-xs text-cyber-accent/50">
                        {session.fileCount} files
                      </div>
                    </div>
                  </div>

                  {/* Quick Output Preview */}
                  {analysis.mainOutputs.length > 0 && (
                    <div className="mb-4">
                      <div className="text-sm font-medium text-cyber-accent/80 mb-2">Key Outputs:</div>
                      <div className="flex flex-wrap gap-2">
                        {analysis.mainOutputs.slice(0, 3).map((file, idx) => (
                          <div key={idx} className="flex items-center gap-1 px-2 py-1 bg-cyber-dark/30 rounded text-xs text-cyber-accent/70">
                            {getFileIcon(file.name)}
                            <span className="truncate max-w-20">{file.name}</span>
                          </div>
                        ))}
                        {analysis.mainOutputs.length > 3 && (
                          <div className="px-2 py-1 bg-cyber-dark/30 rounded text-xs text-cyber-accent/50">
                            +{analysis.mainOutputs.length - 3} more
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Output Type Summary */}
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-4">
                      {analysis.hasCSV && (
                        <div className="flex items-center gap-1 text-green-400">
                          <FileSpreadsheet size={14} />
                          <span>Data</span>
                        </div>
                      )}
                      {analysis.hasPDF && (
                        <div className="flex items-center gap-1 text-blue-400">
                          <FileText size={14} />
                          <span>Resources</span>
                        </div>
                      )}
                      {analysis.hasSummary && (
                        <div className="flex items-center gap-1 text-purple-400">
                          <FileText size={14} />
                          <span>Summary</span>
                        </div>
                      )}
                    </div>
                    <div className="flex items-center gap-1 text-cyber-accent/70">
                      <span>View Details</span>
                      <ArrowRight size={14} />
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        )}
      </motion.div>

      {/* Quick Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className="cyber-card p-6"
      >
        <h3 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
          <Zap size={20} className="text-cyber-accent" />
          Quick Actions
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="cyber-button-outline flex items-center gap-2 justify-center">
            <Download size={16} />
            Export Session Data
          </button>
          <button className="cyber-button-outline flex items-center gap-2 justify-center">
            <BarChart3 size={16} />
            Generate Reports
          </button>
          <button className="cyber-button-outline flex items-center gap-2 justify-center">
            <Settings size={16} />
            Configure Alerts
          </button>
        </div>
      </motion.div>
    </div>
  );
};

export default Dashboard;
