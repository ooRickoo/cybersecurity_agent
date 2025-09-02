import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from 'react-query';
import { 
  ArrowLeft,
  Download,
  FileText,
  FileSpreadsheet,
  FileCode,
  FileImage,
  FileArchive,
  Eye,
  CheckCircle,
  Target,
  Workflow,
  FolderOpen,
  Search
} from 'lucide-react';
import { motion } from 'framer-motion';
import axios from 'axios';

const SessionDetail = () => {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [selectedTab, setSelectedTab] = useState('overview');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');

  // Fetch session data
  const { data: sessionData, isLoading, error } = useQuery(
    ['session', sessionId],
    async () => {
      const response = await axios.get(`/api/sessions/${sessionId}`);
      return response.data;
    }
  );

  const session = sessionData?.session;
  const files = session?.files || [];

  // Analyze session content
  const analyzeSession = () => {
    if (!session) return {};
    
    const mainOutputs = files.filter(f => 
      f.name?.endsWith('.csv') || 
      f.name?.includes('summary') || 
      f.name?.includes('report') ||
      f.name?.includes('analysis')
    );
    
    const resources = files.filter(f => 
      f.name?.endsWith('.pdf') || 
      f.name?.includes('resource') ||
      f.name?.includes('patent_resources')
    );
    
    const workflowFiles = files.filter(f => 
      f.name?.includes('workflow') || 
      f.name?.includes('context') ||
      f.name?.endsWith('.txt')
    );
    
    const dataFiles = files.filter(f => 
      f.name?.endsWith('.csv') || 
      f.name?.endsWith('.json') ||
      f.name?.endsWith('.xlsx')
    );

    return {
      mainOutputs,
      resources,
      workflowFiles,
      dataFiles,
      totalFiles: files.length,
      hasCSV: mainOutputs.some(f => f.name?.endsWith('.csv')),
      hasPDF: resources.some(f => f.name?.endsWith('.pdf')),
      hasSummary: mainOutputs.some(f => f.name?.includes('summary')),
      hasWorkflow: workflowFiles.length > 0
    };
  };

  // Filter files
  const filteredFiles = files.filter(file => {
    const matchesSearch = file.name?.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterType === 'all' || 
                         (filterType === 'outputs' && (file.name?.endsWith('.csv') || file.name?.includes('summary'))) ||
                         (filterType === 'resources' && (file.name?.endsWith('.pdf') || file.name?.includes('resource'))) ||
                         (filterType === 'workflow' && (file.name?.includes('workflow') || file.name?.endsWith('.txt')));
    return matchesSearch && matchesFilter;
  });

  // Get workflow type
  const getWorkflowType = () => {
    if (!session) return 'Unknown';
    const files = session.files || [];
    if (files.some(f => f.name?.includes('patent'))) return 'Patent Analysis';
    if (files.some(f => f.name?.includes('threat'))) return 'Threat Hunting';
    if (files.some(f => f.name?.includes('compliance'))) return 'Compliance';
    if (files.some(f => f.name?.includes('graph'))) return 'Graph Analysis';
    return 'General Analysis';
  };

  // Get file icon
  const getFileIcon = (fileName) => {
    if (fileName?.endsWith('.csv')) return <FileSpreadsheet size={16} />;
    if (fileName?.endsWith('.pdf')) return <FileText size={16} />;
    if (fileName?.endsWith('.json')) return <FileCode size={16} />;
    if (fileName?.match(/\.(png|jpg|jpeg|svg)$/)) return <FileImage size={16} />;
    if (fileName?.match(/\.(zip|tar|gz)$/)) return <FileArchive size={16} />;
    return <FileText size={16} />;
  };

  // Get file type color
  const getFileTypeColor = (fileName) => {
    if (fileName?.endsWith('.csv')) return 'from-green-500 to-teal-600';
    if (fileName?.endsWith('.pdf')) return 'from-blue-500 to-indigo-600';
    if (fileName?.endsWith('.json')) return 'from-purple-500 to-pink-600';
    if (fileName?.match(/\.(png|jpg|jpeg|svg)$/)) return 'from-yellow-500 to-orange-600';
    return 'from-gray-500 to-gray-600';
  };

  // Download file
  const downloadFile = async (filePath) => {
    try {
      // Include session ID in the file path
      const fullPath = `${sessionId}/${filePath}`;
      const response = await axios.get(`/api/download/${encodeURIComponent(fullPath)}`, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filePath.split('/').pop());
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  // View file content
  const viewFile = (filePath) => {
    // Include session ID in the file path
    const fullPath = `${sessionId}/${filePath}`;
    window.open(`/api/view/${encodeURIComponent(fullPath)}`, '_blank');
  };

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
          <p className="text-cyber-accent/70 mb-4">The requested session could not be loaded</p>
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

  const analysis = analyzeSession();
  const workflowType = getWorkflowType();

  return (
    <div className="h-full overflow-y-auto">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-cyber-dark/95 backdrop-blur-sm border-b border-cyber-accent/20">
        <div className="p-6">
          <div className="flex items-center gap-4 mb-4">
            <button
              onClick={() => navigate('/')}
              className="cyber-button-outline p-2"
            >
              <ArrowLeft size={20} />
            </button>
            <div>
              <h1 className="text-2xl font-bold text-white">{session.name}</h1>
              <p className="text-cyber-accent/70 font-mono text-sm">{session.id}</p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex gap-1">
            {[
              { id: 'overview', label: 'Overview', icon: Eye },
              { id: 'outputs', label: 'Outputs', icon: FileText },
              { id: 'workflow', label: 'Workflow', icon: Workflow },
              { id: 'files', label: 'All Files', icon: FolderOpen }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setSelectedTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                  selectedTab === tab.id
                    ? 'bg-cyber-accent text-cyber-dark font-medium'
                    : 'text-cyber-accent/70 hover:text-cyber-accent hover:bg-cyber-accent/10'
                }`}
              >
                <tab.icon size={16} />
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-6 space-y-6">
        {/* Overview Tab */}
        {selectedTab === 'overview' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* Executive Summary */}
            <div className="cyber-card p-6">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <Target size={20} className="text-cyber-accent" />
                Executive Summary
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="text-center p-4 bg-cyber-dark/30 rounded-lg">
                  <div className="text-2xl font-bold text-cyber-accent">{workflowType}</div>
                  <div className="text-sm text-cyber-accent/70">Workflow Type</div>
                </div>
                <div className="text-center p-4 bg-cyber-dark/30 rounded-lg">
                  <div className="text-2xl font-bold text-white">{analysis.totalFiles}</div>
                  <div className="text-sm text-cyber-accent/70">Total Files</div>
                </div>
                <div className="text-center p-4 bg-cyber-dark/30 rounded-lg">
                  <div className="text-2xl font-bold text-green-400">{analysis.mainOutputs.length}</div>
                  <div className="text-sm text-cyber-accent/70">Key Outputs</div>
                </div>
              </div>
              
              {/* Key Outputs Preview */}
              {analysis.mainOutputs.length > 0 && (
                <div>
                  <h3 className="text-lg font-medium text-white mb-3">Key Outputs</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {analysis.mainOutputs.map((file, idx) => (
                      <div key={idx} className="flex items-center gap-3 p-3 bg-cyber-dark/20 rounded-lg">
                        <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${getFileTypeColor(file.name)} flex items-center justify-center`}>
                          {getFileIcon(file.name)}
                        </div>
                        <div className="flex-1">
                          <div className="font-medium text-white">{file.name}</div>
                          <div className="text-sm text-cyber-accent/70">
                            {file.size ? `${(file.size / 1024).toFixed(1)} KB` : 'Unknown size'}
                          </div>
                        </div>
                        <button
                          onClick={() => viewFile(file.path)}
                          className="cyber-button-small"
                        >
                          <Eye size={14} />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Workflow Status */}
            <div className="cyber-card p-6">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <Workflow size={20} className="text-cyber-accent" />
                Workflow Status
              </h2>
              <div className="space-y-3">
                <div className="flex items-center gap-3 p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
                  <CheckCircle size={20} className="text-green-400" />
                  <div>
                    <div className="font-medium text-white">Workflow Completed</div>
                    <div className="text-sm text-cyber-accent/70">All steps executed successfully</div>
                  </div>
                </div>
                {analysis.hasCSV && (
                  <div className="flex items-center gap-3 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                    <FileSpreadsheet size={20} className="text-blue-400" />
                    <div>
                      <div className="font-medium text-white">Data Exported</div>
                      <div className="text-sm text-cyber-accent/70">CSV output generated with results</div>
                    </div>
                  </div>
                )}
                {analysis.hasPDF && (
                  <div className="flex items-center gap-3 p-3 bg-purple-500/10 border border-purple-500/20 rounded-lg">
                    <FileText size={20} className="text-purple-400" />
                    <div>
                      <div className="font-medium text-white">Resources Downloaded</div>
                      <div className="text-sm text-cyber-accent/70">PDF documents and resources available</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}

        {/* Outputs Tab */}
        {selectedTab === 'outputs' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* Main Outputs */}
            {analysis.mainOutputs.length > 0 && (
              <div className="cyber-card p-6">
                <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                  <FileText size={20} className="text-cyber-accent" />
                  Main Outputs
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {analysis.mainOutputs.map((file, idx) => (
                    <div key={idx} className="p-4 bg-cyber-dark/20 rounded-lg border border-cyber-accent/20">
                      <div className="flex items-start gap-3 mb-3">
                        <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${getFileTypeColor(file.name)} flex items-center justify-center`}>
                          {getFileIcon(file.name)}
                        </div>
                        <div className="flex-1">
                          <h3 className="font-medium text-white mb-1">{file.name}</h3>
                          <p className="text-sm text-cyber-accent/70">
                            {file.size ? `${(file.size / 1024).toFixed(1)} KB` : 'Unknown size'}
                          </p>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => viewFile(file.path)}
                          className="cyber-button-small flex-1"
                        >
                          <Eye size={14} className="mr-1" />
                          View
                        </button>
                        <button
                          onClick={() => downloadFile(file.path)}
                          className="cyber-button-small flex-1"
                        >
                          <Download size={14} className="mr-1" />
                          Download
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Resources */}
            {analysis.resources.length > 0 && (
              <div className="cyber-card p-6">
                <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                  <FileText size={20} className="text-cyber-accent" />
                  Resources & Documents
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {analysis.resources.map((file, idx) => (
                    <div key={idx} className="p-4 bg-cyber-dark/20 rounded-lg border border-cyber-accent/20">
                      <div className="flex items-center gap-3 mb-3">
                        <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${getFileTypeColor(file.name)} flex items-center justify-center`}>
                          {getFileIcon(file.name)}
                        </div>
                        <div className="flex-1">
                          <h3 className="font-medium text-white text-sm">{file.name}</h3>
                          <p className="text-xs text-cyber-accent/70">
                            {file.size ? `${(file.size / 1024).toFixed(1)} KB` : 'Unknown size'}
                          </p>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => viewFile(file.path)}
                          className="cyber-button-small flex-1 text-xs"
                        >
                          <Eye size={12} className="mr-1" />
                          View
                        </button>
                        <button
                          onClick={() => downloadFile(file.path)}
                          className="cyber-button-small flex-1 text-xs"
                        >
                          <Download size={12} className="mr-1" />
                          Download
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* Workflow Tab */}
        {selectedTab === 'workflow' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* Workflow Steps */}
            <div className="cyber-card p-6">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <Workflow size={20} className="text-cyber-accent" />
                Workflow Execution Steps
              </h2>
              <div className="space-y-4">
                <div className="flex items-center gap-4 p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                  <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center">
                    <CheckCircle size={16} className="text-white" />
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-white">Input Processing</div>
                    <div className="text-sm text-cyber-accent/70">CSV file loaded and validated</div>
                  </div>
                  <div className="text-xs text-cyber-accent/50">Step 1</div>
                </div>

                <div className="flex items-center gap-4 p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                  <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center">
                    <CheckCircle size={16} className="text-white" />
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-white">Patent Data Retrieval</div>
                    <div className="text-sm text-cyber-accent/70">USPTO and Google Patents APIs queried</div>
                  </div>
                  <div className="text-xs text-cyber-accent/50">Step 2</div>
                </div>

                <div className="flex items-center gap-4 p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                  <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center">
                    <CheckCircle size={16} className="text-white" />
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-white">LLM Analysis</div>
                    <div className="text-sm text-cyber-accent/70">OpenAI analysis of patent data</div>
                  </div>
                  <div className="text-xs text-cyber-accent/50">Step 3</div>
                </div>

                <div className="flex items-center gap-4 p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                  <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center">
                    <CheckCircle size={16} className="text-white" />
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-white">Output Generation</div>
                    <div className="text-sm text-cyber-accent/70">CSV export and summary artifacts created</div>
                  </div>
                  <div className="text-xs text-cyber-accent/50">Step 4</div>
                </div>
              </div>
            </div>

            {/* Workflow Files */}
            {analysis.workflowFiles.length > 0 && (
              <div className="cyber-card p-6">
                <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                  <FileCode size={20} className="text-cyber-accent" />
                  Workflow Configuration & Logs
                </h2>
                <div className="space-y-3">
                  {analysis.workflowFiles.map((file, idx) => (
                    <div key={idx} className="flex items-center gap-3 p-3 bg-cyber-dark/20 rounded-lg">
                      <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${getFileTypeColor(file.name)} flex items-center justify-center`}>
                        {getFileIcon(file.name)}
                      </div>
                      <div className="flex-1">
                        <div className="font-medium text-white">{file.name}</div>
                        <div className="text-sm text-cyber-accent/70">
                          {file.size ? `${(file.size / 1024).toFixed(1)} KB` : 'Unknown size'}
                        </div>
                      </div>
                      <button
                        onClick={() => viewFile(file.path)}
                        className="cyber-button-small"
                      >
                        <Eye size={14} />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* Files Tab */}
        {selectedTab === 'files' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* Search and Filter */}
            <div className="cyber-card p-4">
              <div className="flex flex-col md:flex-row gap-4 items-center">
                <div className="flex-1 relative">
                  <Search size={20} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-cyber-accent/50" />
                  <input
                    type="text"
                    placeholder="Search files by name..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 bg-cyber-dark/50 border border-cyber-accent/30 rounded-lg text-white placeholder-cyber-accent/50 focus:border-cyber-accent focus:outline-none"
                  />
                </div>
                <select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value)}
                  className="px-4 py-2 bg-cyber-dark/50 border border-cyber-accent/30 rounded-lg text-white focus:border-cyber-accent focus:outline-none"
                >
                  <option value="all">All Files</option>
                  <option value="outputs">Outputs</option>
                  <option value="resources">Resources</option>
                  <option value="workflow">Workflow</option>
                </select>
              </div>
            </div>

            {/* Files List */}
            <div className="cyber-card p-6">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <FolderOpen size={20} className="text-cyber-accent" />
                All Files ({filteredFiles.length})
              </h2>
              {filteredFiles.length === 0 ? (
                <div className="text-center py-8">
                  <div className="text-cyber-accent/50 text-4xl mb-2">📁</div>
                  <p className="text-cyber-accent/70">No files match your search criteria</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {filteredFiles.map((file, idx) => (
                    <div key={idx} className="flex items-center gap-3 p-3 bg-cyber-dark/20 rounded-lg border border-cyber-accent/20">
                      <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${getFileTypeColor(file.name)} flex items-center justify-center`}>
                        {getFileIcon(file.name)}
                      </div>
                      <div className="flex-1">
                        <div className="font-medium text-white">{file.name}</div>
                        <div className="text-sm text-cyber-accent/70">
                          {file.size ? `${(file.size / 1024).toFixed(1)} KB` : 'Unknown size'}
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => viewFile(file.path)}
                          className="cyber-button-small"
                          title="View file"
                        >
                          <Eye size={14} />
                        </button>
                        <button
                          onClick={() => downloadFile(file.path)}
                          className="cyber-button-small"
                          title="Download file"
                        >
                          <Download size={14} />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default SessionDetail;
