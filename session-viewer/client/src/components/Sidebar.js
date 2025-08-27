import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  X, 
  Home, 
  FolderOpen, 
  Settings, 
  Shield,
  BarChart3,
  FileText,
  Database,
  Image,
  Archive
} from 'lucide-react';

const Sidebar = ({ isOpen, onClose, isConnected }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const navigationItems = [
    {
      name: 'Dashboard',
      path: '/',
      icon: Home,
      description: 'Overview and analytics'
    },
    {
      name: 'Sessions',
      path: '/sessions',
      icon: FolderOpen,
      description: 'Browse workflow sessions'
    },
    {
      name: 'Files',
      path: '/files',
      icon: FileText,
      description: 'File management'
    },
    {
      name: 'Analytics',
      path: '/analytics',
      icon: BarChart3,
      description: 'Data analysis tools'
    },
    {
      name: 'Settings',
      path: '/settings',
      icon: Settings,
      description: 'Configuration options'
    }
  ];

  const fileTypeItems = [
    { name: 'Images', icon: Image, count: 0, color: 'from-blue-500 to-purple-600' },
    { name: 'Documents', icon: FileText, count: 0, color: 'from-green-500 to-teal-600' },
    { name: 'Data Files', icon: BarChart3, count: 0, color: 'from-orange-500 to-red-600' },
    { name: 'Databases', icon: Database, count: 0, color: 'from-indigo-500 to-blue-600' },
    { name: 'Archives', icon: Archive, count: 0, color: 'from-purple-500 to-pink-600' }
  ];

  const handleNavigation = (path) => {
    navigate(path);
    onClose();
  };

  return (
    <>
      {/* Backdrop */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 z-50 w-64 bg-cyber-light/95 backdrop-blur-md border-r border-cyber-accent/20
        transform transition-transform duration-300 ease-in-out lg:translate-x-0
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-cyber-accent/20">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyber-accent to-cyber-accent3 flex items-center justify-center">
              <Shield size={20} className="text-cyber-dark" />
            </div>
            <div>
              <h1 className="text-lg font-bold cyber-gradient-text">Cybersecurity Agent</h1>
              <p className="text-xs text-cyber-accent/70">Session Viewer</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="lg:hidden p-2 rounded-lg bg-cyber-light/60 hover:bg-cyber-light/80 text-cyber-accent transition-colors duration-200"
          >
            <X size={20} />
          </button>
        </div>

        {/* Navigation */}
        <nav className="p-4 space-y-2">
          <div className="text-xs font-semibold text-cyber-accent/50 uppercase tracking-wider mb-3">
            Navigation
          </div>
          
          {navigationItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            
            return (
              <button
                key={item.path}
                onClick={() => handleNavigation(item.path)}
                className={`
                  w-full flex items-center gap-3 p-3 rounded-lg text-left transition-all duration-200
                  ${isActive 
                    ? 'bg-cyber-accent/20 text-cyber-accent border border-cyber-accent/30' 
                    : 'text-cyber-accent/70 hover:text-cyber-accent hover:bg-cyber-light/40'
                  }
                `}
              >
                <Icon size={18} />
                <div className="flex-1">
                  <div className="font-medium">{item.name}</div>
                  <div className="text-xs opacity-70">{item.description}</div>
                </div>
              </button>
            );
          })}
        </nav>

        {/* File Types */}
        <div className="p-4 border-t border-cyber-accent/20">
          <div className="text-xs font-semibold text-cyber-accent/50 uppercase tracking-wider mb-3">
            File Types
          </div>
          
          <div className="space-y-2">
            {fileTypeItems.map((item) => {
              const Icon = item.icon;
              
              return (
                <div
                  key={item.name}
                  className="flex items-center gap-3 p-2 rounded-lg hover:bg-cyber-light/40 transition-colors duration-200 cursor-pointer"
                >
                  <div className={`w-6 h-6 rounded bg-gradient-to-br ${item.color} flex items-center justify-center`}>
                    <Icon size={12} className="text-white" />
                  </div>
                  <div className="flex-1 text-sm text-cyber-accent/70">{item.name}</div>
                  <div className="text-xs text-cyber-accent/50">{item.count}</div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Status */}
        <div className="p-4 border-t border-cyber-accent/20 mt-auto">
          <div className="flex items-center gap-2 p-3 rounded-lg bg-cyber-light/40 border border-cyber-accent/20">
            <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}></div>
            <div className="flex-1">
              <div className="text-sm font-medium text-cyber-accent/70">
                {isConnected ? 'Connected' : 'Disconnected'}
              </div>
              <div className="text-xs text-cyber-accent/50">
                {isConnected ? 'Server active' : 'Server offline'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Sidebar;
