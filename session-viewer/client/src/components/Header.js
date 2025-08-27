import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  Menu, 
  Shield, 
  Activity, 
  Home,
  FolderOpen,
  Settings
} from 'lucide-react';

const Header = ({ onMenuClick, isConnected }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const getPageTitle = () => {
    if (location.pathname === '/') return 'Dashboard';
    if (location.pathname.startsWith('/sessions/')) return 'Session Viewer';
    if (location.pathname.startsWith('/files/')) return 'File Viewer';
    return 'Session Viewer';
  };

  const getBreadcrumbs = () => {
    const paths = location.pathname.split('/').filter(Boolean);
    const breadcrumbs = [{ name: 'Home', path: '/' }];
    
    paths.forEach((path, index) => {
      if (path === 'sessions') {
        breadcrumbs.push({ name: 'Sessions', path: '/sessions' });
      } else if (path === 'files') {
        breadcrumbs.push({ name: 'Files', path: '/files' });
      } else if (index > 0) {
        breadcrumbs.push({ name: path, path: location.pathname });
      }
    });
    
    return breadcrumbs;
  };

  return (
    <header className="bg-cyber-light/80 backdrop-blur-sm border-b border-cyber-accent/20 z-20">
      <div className="flex items-center justify-between px-4 py-3">
        {/* Left Section */}
        <div className="flex items-center gap-4">
          {/* Mobile Menu Button */}
          <button
            onClick={onMenuClick}
            className="lg:hidden p-2 rounded-lg bg-cyber-light/60 hover:bg-cyber-light/80 text-cyber-accent transition-colors duration-200"
            aria-label="Open menu"
          >
            <Menu size={20} />
          </button>

          {/* Logo and Title */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyber-accent to-cyber-accent3 flex items-center justify-center">
              <Shield size={20} className="text-cyber-dark" />
            </div>
            <div className="hidden sm:block">
              <h1 className="text-lg font-bold cyber-gradient-text">
                Cybersecurity Agent
              </h1>
              <p className="text-xs text-cyber-accent/70">
                Session Viewer
              </p>
            </div>
          </div>

          {/* Page Title */}
          <div className="hidden md:block ml-6">
            <h2 className="text-lg font-semibold text-white">
              {getPageTitle()}
            </h2>
          </div>
        </div>

        {/* Center Section - Breadcrumbs */}
        <div className="hidden lg:flex items-center gap-2">
          {getBreadcrumbs().map((crumb, index) => (
            <React.Fragment key={crumb.path}>
              {index > 0 && (
                <span className="text-cyber-accent/50 mx-2">
                  /
                </span>
              )}
              <button
                onClick={() => navigate(crumb.path)}
                className={`px-3 py-1 rounded-lg text-sm font-medium transition-all duration-200 ${
                  location.pathname === crumb.path
                    ? 'bg-cyber-accent/20 text-cyber-accent border border-cyber-accent/30'
                    : 'text-cyber-accent/70 hover:text-cyber-accent hover:bg-cyber-light/40'
                }`}
              >
                {crumb.name}
              </button>
            </React.Fragment>
          ))}
        </div>

        {/* Right Section */}
        <div className="flex items-center gap-3">
          {/* Connection Status */}
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-cyber-light/40 border border-cyber-accent/20">
            <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}></div>
            <span className="text-sm font-medium text-cyber-accent/70">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          {/* Quick Actions */}
          <div className="hidden sm:flex items-center gap-2">
            <button
              onClick={() => navigate('/')}
              className="p-2 rounded-lg bg-cyber-light/60 hover:bg-cyber-light/80 text-cyber-accent transition-colors duration-200"
              title="Dashboard"
            >
              <Home size={18} />
            </button>
            
            <button
              onClick={() => navigate('/sessions')}
              className="p-2 rounded-lg bg-cyber-light/60 hover:bg-cyber-light/80 text-cyber-accent transition-colors duration-200"
              title="Sessions"
            >
              <FolderOpen size={18} />
            </button>
            
            <button
              className="p-2 rounded-lg bg-cyber-light/60 hover:bg-cyber-light/80 text-cyber-accent transition-colors duration-200"
              title="Settings"
            >
              <Settings size={18} />
            </button>
          </div>

          {/* System Status */}
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-cyber-light/40 border border-cyber-accent/20">
            <Activity size={16} className="text-cyber-accent" />
            <span className="text-sm font-medium text-cyber-accent/70">
              Live
            </span>
          </div>
        </div>
      </div>

      {/* Mobile Breadcrumbs */}
      <div className="lg:hidden px-4 pb-3">
        <div className="flex items-center gap-2 overflow-x-auto">
          {getBreadcrumbs().map((crumb, index) => (
            <React.Fragment key={crumb.path}>
              {index > 0 && (
                <span className="text-cyber-accent/50 text-xs">
                  ›
                </span>
              )}
              <button
                onClick={() => navigate(crumb.path)}
                className={`px-2 py-1 rounded text-xs font-medium whitespace-nowrap transition-all duration-200 ${
                  location.pathname === crumb.path
                    ? 'bg-cyber-accent/20 text-cyber-accent'
                    : 'text-cyber-accent/70 hover:text-cyber-accent'
                }`}
              >
                {crumb.name}
              </button>
            </React.Fragment>
          ))}
        </div>
      </div>
    </header>
  );
};

export default Header;
