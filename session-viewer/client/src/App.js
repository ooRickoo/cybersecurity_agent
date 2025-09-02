import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';
import io from 'socket.io-client';

// Components
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import SessionDetail from './components/SessionDetail';
import FileViewer from './components/FileViewer';
import StatusBar from './components/StatusBar';

// Context
import { SessionProvider } from './context/SessionContext';
import { SocketProvider } from './components/SocketContext';

// Styles
import './index.css';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Initialize Socket.IO connection
    const newSocket = io('http://localhost:3001', {
      transports: ['websocket', 'polling'],
      timeout: 20000,
    });

    newSocket.on('connect', () => {
      console.log('🔌 Connected to session viewer server');
      setIsConnected(true);
    });

    newSocket.on('disconnect', () => {
      console.log('🔌 Disconnected from session viewer server');
      setIsConnected(false);
    });

    newSocket.on('connect_error', (error) => {
      console.error('🔌 Connection error:', error);
      setIsConnected(false);
    });

    setSocket(newSocket);

    // Hide loading screen when React app is ready
    if (window.hideLoadingScreen) {
      setTimeout(() => {
        window.hideLoadingScreen();
      }, 100);
    }

    // Cleanup on unmount
    return () => {
      if (newSocket) {
        newSocket.disconnect();
      }
    };
  }, []);

  // Auto-close sidebar on mobile when route changes
  const handleRouteChange = () => {
    if (window.innerWidth < 1024) {
      setSidebarOpen(false);
    }
  };

  return (
    <QueryClientProvider client={queryClient}>
      <SocketProvider socket={socket} isConnected={isConnected}>
        <SessionProvider>
          <Router>
            <div className="min-h-screen bg-cyber-dark text-white">
              {/* Background Pattern */}
              <div className="fixed inset-0 bg-cyber-darker">
                <div className="absolute inset-0 bg-gradient-to-br from-cyber-dark via-cyber-light to-cyber-dark opacity-20"></div>
                <div className="absolute inset-0 opacity-30">
                  <svg className="w-full h-full" width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                      <pattern id="grid" width="60" height="60" patternUnits="userSpaceOnUse">
                        <circle cx="30" cy="30" r="1" fill="#00d4aa" fillOpacity="0.03"/>
                      </pattern>
                    </defs>
                    <rect width="100%" height="100%" fill="url(#grid)" />
                  </svg>
                </div>
              </div>

              {/* Main Content */}
              <div className="relative z-10 flex h-screen">
                {/* Sidebar */}
                <Sidebar 
                  isOpen={sidebarOpen} 
                  onClose={() => setSidebarOpen(false)}
                  isConnected={isConnected}
                  collapsed={sidebarCollapsed}
                />

                {/* Main Content Area */}
                <div className="flex-1 flex flex-col overflow-hidden">
                  {/* Header */}
                  <Header 
                    onMenuClick={() => setSidebarOpen(true)}
                    isConnected={isConnected}
                    sidebarOpen={sidebarOpen}
                    onSidebarToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
                  />

                  {/* Main Content */}
                  <main className="flex-1 overflow-hidden">
                    <Routes>
                      <Route 
                        path="/" 
                        element={<Dashboard onRouteChange={handleRouteChange} />} 
                      />
                      <Route 
                        path="/session/:sessionId" 
                        element={<SessionDetail onRouteChange={handleRouteChange} />} 
                      />
                      <Route 
                        path="/files/:filePath" 
                        element={<FileViewer onRouteChange={handleRouteChange} />} 
                      />
                      <Route 
                        path="*" 
                        element={<Navigate to="/" replace />} 
                      />
                    </Routes>
                  </main>

                  {/* Status Bar */}
                  <StatusBar isConnected={isConnected} />
                </div>
              </div>

              {/* Toast Notifications */}
              <Toaster
                position="top-right"
                toastOptions={{
                  duration: 4000,
                  style: {
                    background: '#1a1a2e',
                    color: '#ffffff',
                    border: '1px solid #00d4aa',
                  },
                  success: {
                    iconTheme: {
                      primary: '#00d4aa',
                      secondary: '#ffffff',
                    },
                  },
                  error: {
                    iconTheme: {
                      primary: '#ff6b6b',
                      secondary: '#ffffff',
                    },
                  },
                }}
              />
            </div>
          </Router>
        </SessionProvider>
      </SocketProvider>
    </QueryClientProvider>
  );
}

export default App;
