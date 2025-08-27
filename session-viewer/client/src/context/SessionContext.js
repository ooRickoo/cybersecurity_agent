import React, { createContext, useContext, useReducer } from 'react';

const SessionContext = createContext();

const initialState = {
  sessions: [],
  currentSession: null,
  selectedFiles: [],
  isLoading: false,
  error: null,
  filters: {
    fileType: 'all',
    dateRange: 'all',
    searchQuery: ''
  }
};

const sessionReducer = (state, action) => {
  switch (action.type) {
    case 'SET_SESSIONS':
      return {
        ...state,
        sessions: action.payload,
        isLoading: false,
        error: null
      };
    
    case 'SET_CURRENT_SESSION':
      return {
        ...state,
        currentSession: action.payload
      };
    
    case 'ADD_SELECTED_FILE':
      return {
        ...state,
        selectedFiles: [...state.selectedFiles, action.payload]
      };
    
    case 'REMOVE_SELECTED_FILE':
      return {
        ...state,
        selectedFiles: state.selectedFiles.filter(file => file.path !== action.payload.path)
      };
    
    case 'CLEAR_SELECTED_FILES':
      return {
        ...state,
        selectedFiles: []
      };
    
    case 'SET_LOADING':
      return {
        ...state,
        isLoading: action.payload
      };
    
    case 'SET_ERROR':
      return {
        ...state,
        error: action.payload,
        isLoading: false
      };
    
    case 'SET_FILTERS':
      return {
        ...state,
        filters: {
          ...state.filters,
          ...action.payload
        }
      };
    
    case 'RESET_FILTERS':
      return {
        ...state,
        filters: initialState.filters
      };
    
    default:
      return state;
  }
};

export const SessionProvider = ({ children }) => {
  const [state, dispatch] = useReducer(sessionReducer, initialState);

  const value = {
    ...state,
    dispatch,
    actions: {
      setSessions: (sessions) => dispatch({ type: 'SET_SESSIONS', payload: sessions }),
      setCurrentSession: (session) => dispatch({ type: 'SET_CURRENT_SESSION', payload: session }),
      addSelectedFile: (file) => dispatch({ type: 'ADD_SELECTED_FILE', payload: file }),
      removeSelectedFile: (file) => dispatch({ type: 'REMOVE_SELECTED_FILE', payload: file }),
      clearSelectedFiles: () => dispatch({ type: 'CLEAR_SELECTED_FILES' }),
      setLoading: (loading) => dispatch({ type: 'SET_LOADING', payload: loading }),
      setError: (error) => dispatch({ type: 'SET_ERROR', payload: error }),
      setFilters: (filters) => dispatch({ type: 'SET_FILTERS', payload: filters }),
      resetFilters: () => dispatch({ type: 'RESET_FILTERS' })
    }
  };

  return (
    <SessionContext.Provider value={value}>
      {children}
    </SessionContext.Provider>
  );
};

export const useSession = () => {
  const context = useContext(SessionContext);
  if (!context) {
    throw new Error('useSession must be used within a SessionProvider');
  }
  return context;
};
