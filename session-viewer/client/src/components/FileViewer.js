import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from 'react-query';
import { 
  ArrowLeft, 
  Download, 
  FileText, 
  Image, 
  Database,
  Archive,
  Code,
  Eye
} from 'lucide-react';
import { motion } from 'framer-motion';
import axios from 'axios';

const FileViewer = ({ onRouteChange }) => {
  const { filePath } = useParams();
  const navigate = useNavigate();
  const [fileContent, setFileContent] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Fetch file content
  const { error } = useQuery(
    ['file', filePath],
    async () => {
      const response = await axios.get(`/api/files/${encodeURIComponent(filePath)}`);
      return response.data;
    },
    {
      onSuccess: (data) => {
        setFileContent(data);
        setIsLoading(false);
      },
      onError: () => {
        setIsLoading(false);
      }
    }
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="cyber-spinner"></div>
      </div>
    );
  }

  if (error || !fileContent) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="text-accent-500 text-6xl mb-4">⚠️</div>
          <h2 className="text-xl font-semibold text-white mb-2">File Not Found</h2>
          <p className="text-cyber-accent/70 mb-4">
            {error?.message || 'The requested file could not be loaded.'}
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
      case 'image': return <Image size={20} />;
      case 'document': return <FileText size={20} />;
      case 'data': return <Code size={20} />;
      case 'database': return <Database size={20} />;
      case 'archive': return <Archive size={20} />;
      default: return <FileText size={20} />;
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

  const renderFileContent = () => {
    // eslint-disable-next-line no-unused-vars
    const { content, type, stats } = fileContent;

    switch (type) {
      case 'image':
        return (
          <div className="text-center">
            <img 
              src={`/api/files/${encodeURIComponent(filePath)}`} 
              alt={filePath.split('/').pop()}
              className="max-w-full max-h-[70vh] rounded-lg shadow-cyber"
            />
          </div>
        );

      case 'csv':
        return (
          <div className="overflow-x-auto">
            <pre className="text-sm font-mono text-cyber-accent/90 bg-cyber-light/20 p-4 rounded-lg overflow-x-auto">
              {content}
            </pre>
          </div>
        );

      case 'json':
        return (
          <div className="overflow-x-auto">
            <pre className="text-sm font-mono text-cyber-accent/90 bg-cyber-light/20 p-4 rounded-lg overflow-x-auto">
              {JSON.stringify(content, null, 2)}
            </pre>
          </div>
        );

      case 'markdown':
        return (
          <div className="prose prose-invert max-w-none">
            <div className="text-sm text-cyber-accent/90 bg-cyber-light/20 p-4 rounded-lg">
              {content}
            </div>
          </div>
        );

      default:
        return (
          <div className="text-center py-8">
            <div className="text-cyber-accent/50 text-6xl mb-4">📄</div>
            <h3 className="text-lg font-semibold text-white mb-2">File Preview Not Available</h3>
            <p className="text-cyber-accent/70 mb-4">
              This file type ({type}) cannot be previewed in the browser.
            </p>
            <button
              onClick={() => {
                const downloadUrl = `/api/download/${encodeURIComponent(filePath)}`;
                window.open(downloadUrl, '_blank');
              }}
              className="cyber-button"
            >
              <Download size={16} className="mr-2" />
              Download File
            </button>
          </div>
        );
    }
  };

  const fileName = filePath.split('/').pop();
  const fileType = fileContent.type || 'unknown';

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
            <div className="flex items-center gap-3">
              <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${getFileTypeColor(fileType)} flex items-center justify-center`}>
                {getFileIcon(fileType)}
              </div>
              <div>
                <div className="cyber-panel-title">
                  {fileName}
                </div>
                <div className="cyber-panel-subtitle">
                  {filePath} • {fileType} • {fileContent.stats?.size ? `${(fileContent.stats.size / 1024).toFixed(1)} KB` : 'Unknown size'}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* File Actions */}
        <div className="flex gap-3">
          <button
            onClick={() => {
              const downloadUrl = `/api/download/${encodeURIComponent(filePath)}`;
              window.open(downloadUrl, '_blank');
            }}
            className="cyber-button"
          >
            <Download size={16} className="mr-2" />
            Download
          </button>
          
          <button
            onClick={() => navigate('/')}
            className="cyber-button-secondary"
          >
            <Eye size={16} className="mr-2" />
            Browse Files
          </button>
        </div>
      </motion.div>

      {/* File Content */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="cyber-panel"
      >
        <div className="cyber-panel-header">
          <div className="cyber-panel-title">
            <Eye className="text-cyber-accent" />
            File Preview
          </div>
          <div className="cyber-panel-subtitle">
            {fileType === 'image' ? 'Image display' : 
             fileType === 'csv' ? 'CSV data view' :
             fileType === 'json' ? 'JSON structure' :
             fileType === 'markdown' ? 'Markdown content' :
             'File information'}
          </div>
        </div>

        <div className="min-h-[400px]">
          {renderFileContent()}
        </div>
      </motion.div>

      {/* File Metadata */}
      {fileContent.stats && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="cyber-panel"
        >
          <div className="cyber-panel-header">
            <div className="cyber-panel-title">
              <FileText className="text-cyber-accent" />
              File Information
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="cyber-card p-3 text-center">
              <div className="text-sm text-cyber-accent/70">Type</div>
              <div className="font-semibold text-white capitalize">{fileType}</div>
            </div>
            <div className="cyber-card p-3 text-center">
              <div className="text-sm text-cyber-accent/70">Size</div>
              <div className="font-semibold text-white">
                {fileContent.stats.size ? `${(fileContent.stats.size / 1024).toFixed(1)} KB` : 'Unknown'}
              </div>
            </div>
            <div className="cyber-card p-3 text-center">
              <div className="text-sm text-cyber-accent/70">Modified</div>
              <div className="font-semibold text-white">
                {new Date(fileContent.stats.mtime).toLocaleString()}
              </div>
            </div>
            <div className="cyber-card p-3 text-center">
              <div className="text-sm text-cyber-accent/70">Path</div>
              <div className="font-semibold text-white text-xs truncate">{filePath}</div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default FileViewer;
