import React, { createContext, useContext, useEffect, useState } from 'react';

const SocketContext = createContext();

export const SocketProvider = ({ children, socket, isConnected }) => {
  const [socketInstance, setSocketInstance] = useState(socket);
  const [connectionStatus, setConnectionStatus] = useState({
    isConnected: false,
    isConnecting: false,
    error: null
  });

  useEffect(() => {
    if (socket) {
      setSocketInstance(socket);
      
      // Set up event listeners
      socket.on('connect', () => {
        setConnectionStatus({
          isConnected: true,
          isConnecting: false,
          error: null
        });
      });

      socket.on('disconnect', () => {
        setConnectionStatus({
          isConnected: false,
          isConnecting: false,
          error: null
        });
      });

      socket.on('connect_error', (error) => {
        setConnectionStatus({
          isConnected: false,
          isConnecting: false,
          error: error.message
        });
      });

      // File system events
      socket.on('fileAdded', (data) => {
        console.log('📁 File added:', data);
        // You can emit custom events or use a state management solution here
      });

      socket.on('fileChanged', (data) => {
        console.log('📝 File changed:', data);
      });

      socket.on('fileRemoved', (data) => {
        console.log('🗑️ File removed:', data);
      });

      socket.on('sessionsUpdate', (sessions) => {
        console.log('🔄 Sessions updated:', sessions);
      });

      socket.on('error', (error) => {
        console.error('🔌 Socket error:', error);
        setConnectionStatus(prev => ({
          ...prev,
          error: error.message
        }));
      });

      // Cleanup on unmount
      return () => {
        socket.off('connect');
        socket.off('disconnect');
        socket.off('connect_error');
        socket.off('fileAdded');
        socket.off('fileChanged');
        socket.off('fileRemoved');
        socket.off('sessionsUpdate');
        socket.off('error');
      };
    }
  }, [socket]);

  const value = {
    socket: socketInstance,
    connectionStatus,
    emit: (event, data) => {
      if (socketInstance && connectionStatus.isConnected) {
        socketInstance.emit(event, data);
      }
    }
  };

  return (
    <SocketContext.Provider value={value}>
      {children}
    </SocketContext.Provider>
  );
};

export const useSocket = () => {
  const context = useContext(SocketContext);
  if (!context) {
    throw new Error('useSocket must be used within a SocketProvider');
  }
  return context;
};
