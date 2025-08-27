import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from 'react-query';
import { 
  ArrowLeft, 
  Download, 
  Eye, 
  FileText, 
  Image, 
  Database,
  Archive,
  Clock,
  HardDrive,
  FolderOpen
} from 'lucide-react';
import { motion } from 'framer-motion';
import axios from 'axios';

const SessionViewer = ({ onRouteChange }) => {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);

  // Fetch session data
  const { data: sessionData, isLoading, error } = useQuery(
    ['session', sessionId],
    async () => {
      const response = await axios.get(`/api/sessions/${sessionId}`);
      return response.data;
    },
    {
      refetchInterval: 5000, // Refresh every 5 seconds
    }
  );

  const session = sessionData?.session;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="cyber-spinner"></div>
      </div>
    );
  }

  if (error || !session) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="text-accent-500 text-6xl mb-4">⚠️</div>
          <h2 className="text-xl font-semibold text-white mb-2">Session Not Found</h2>
          <p className="text-cyber-accent/70 mb-4">
            {error?.message || 'The requested session could not be found.'}
          </p>
          <button 
            onClick={() => navigate('/')}
            className="cyber-button"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const getFileIcon = (type) => {
    switch (type) {
      case 'image': return <Image size={16} />;
      case 'document': return <FileText size={16} />;
      case 'data': return <FileText size={16} />;
      case 'database': return <Database size={16} />;
      case 'archive': return <Archive size={16} />;
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

  const handleFileClick = (file) => {
    setSelectedFile(file);
  };

  const handleDownload = (file) => {
    const downloadUrl = `/api/download/${encodeURIComponent(file.path)}`;
    window.open(downloadUrl, '_blank');
  };

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="cyber-panel"
      >
        <div className="cyber-panel-header">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/')}
              className="p-2 rounded-lg bg-cyber-light/60 hover:bg-cyber-light/80 text-cyber-accent transition-colors duration-200"
            >
              <ArrowLeft size={20} />
            </button>
            <div>
              <div className="cyber-panel-title">
                <FolderOpen className="text-cyber-accent" />
                {session.name}
              </div>
              <div className="cyber-panel-subtitle">
                Session ID: {session.id} • {session.fileCount} files • {session.totalSizeFormatted}
              </div>
            </div>
          </div>
        </div>

        {/* Session Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="cyber-card cyber-card-hover p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyber-accent to-cyber-accent3 flex items-center justify-center">
                <FileText size={20} className="text-cyber-dark" />
              </div>
              <div>
                <div className="text-2xl font-bold text-white">{session.fileCount}</div>
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
                <div className="text-2xl font-bold text-white">{session.totalSizeFormatted}</div>
                <div className="text-sm text-cyber-accent/70">Total Size</div>
              </div>
            </div>
          </div>

          <div className="cyber-card cyber-card-hover p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center">
                <Clock size={20} className="text-white" />
              </div>
              <div>
                <div className="text-2xl font-bold text-white">
                  {new Date(session.lastModified).toLocaleDateString()}
                </div>
                <div className="text-sm text-cyber-accent/70">Last Modified</div>
              </div>
            </div>
          </div>

          <div className="cyber-card cyber-card-hover p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-green-500 to-teal-600 flex items-center justify-center">
                <FolderOpen size={20} className="text-white" />
              </div>
              <div>
                <div className="text-2xl font-bold text-white">{session.id}</div>
                <div className="text-sm text-cyber-accent/70">Session ID</div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Files List */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="cyber-panel"
      >
        <div className="cyber-panel-header">
          <div className="cyber-panel-title">
            <FileText className="text-cyber-accent" />
            Session Files
          </div>
          <div className="cyber-panel-subtitle">
            Browse and interact with workflow output files
          </div>
        </div>

        {session.files && session.files.length > 0 ? (
          <div className="cyber-list">
            {session.files.map((file, index) => (
              <motion.div
                key={file.path}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                className="cyber-list-item cursor-pointer"
                onClick={() => handleFileClick(file)}
              >
                <div className={`cyber-list-item-icon ${getFileTypeColor(file.type)}`}>
                  {getFileIcon(file.type)}
                </div>
                <div className="cyber-list-item-content">
                  <div className="cyber-list-item-title">{file.name}</div>
                  <div className="cyber-list-item-subtitle">
                    {file.sizeFormatted} • {file.type} • {new Date(file.lastModified).toLocaleString()}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDownload(file);
                    }}
                    className="cyber-button-secondary text-xs"
                    title="Download file"
                  >
                    <Download size={14} />
                  </button>
                  <button
                    className="cyber-button-secondary text-xs"
                    title="View file"
                  >
                    <Eye size={14} />
                  </button>
                </div>
              </motion.div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <div className="text-cyber-accent/50 text-6xl mb-4">📁</div>
            <h3 className="text-lg font-semibold text-white mb-2">No Files Found</h3>
            <p className="text-cyber-accent/70">This session doesn't contain any output files yet.</p>
          </div>
        )}
      </motion.div>

      {/* File Preview Modal */}
      {selectedFile && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="cyber-panel max-w-4xl max-h-[90vh] overflow-y-auto">
            <div className="cyber-panel-header">
              <div className="flex items-center justify-between">
                <div className="cyber-panel-title">
                  {getFileIcon(selectedFile.type)}
                  {selectedFile.name}
                </div>
                <button
                  onClick={() => setSelectedFile(null)}
                  className="p-2 rounded-lg bg-cyber-light/60 hover:bg-cyber-light/80 text-cyber-accent transition-colors duration-200"
                >
                  <ArrowLeft size={20} />
                </button>
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="cyber-card p-3 text-center">
                  <div className="text-sm text-cyber-accent/70">Type</div>
                  <div className="font-semibold text-white capitalize">{selectedFile.type}</div>
                </div>
                <div className="cyber-card p-3 text-center">
                  <div className="text-sm text-cyber-accent/70">Size</div>
                  <div className="font-semibold text-white">{selectedFile.sizeFormatted}</div>
                </div>
                <div className="cyber-card p-3 text-center">
                  <div className="text-sm text-cyber-accent/70">Modified</div>
                  <div className="font-semibold text-white">
                    {new Date(selectedFile.lastModified).toLocaleDateString()}
                  </div>
                </div>
                <div className="cyber-card p-3 text-center">
                  <div className="text-sm text-cyber-accent/70">Path</div>
                  <div className="font-semibold text-white text-xs truncate">{selectedFile.path}</div>
                </div>
              </div>
              
              <div className="flex gap-2">
                <button
                  onClick={() => handleDownload(selectedFile)}
                  className="cyber-button"
                >
                  <Download size={16} className="mr-2" />
                  Download
                </button>
                <button
                  onClick={() => setSelectedFile(null)}
                  className="cyber-button-secondary"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SessionViewer;
