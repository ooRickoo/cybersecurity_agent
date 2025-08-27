#!/usr/bin/env node
/**
 * Cybersecurity Agent Session Viewer Server
 * Provides a professional web interface for viewing workflow outputs
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const morgan = require('morgan');
const path = require('path');
const fs = require('fs-extra');
const chokidar = require('chokidar');
const { createServer } = require('http');
const { Server } = require('socket.io');
const sqlite3 = require('sqlite3').verbose();

const app = express();
const server = createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

// Configuration
const PORT = process.env.PORT || 3001;
const SESSION_OUTPUTS_DIR = path.join(__dirname, '..', 'session-outputs');
const KNOWLEDGE_OBJECTS_DIR = path.join(__dirname, '..', 'knowledge-objects');
const SAMPLE_DATA_DIR = path.join(__dirname, 'data');

// Middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "blob:"],
      connectSrc: ["'self'", "ws:", "wss:"]
    }
  }
}));
app.use(compression());
app.use(cors());
app.use(morgan('combined'));
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'client', 'build')));

// File watcher for real-time updates
let fileWatcher = null;

function initializeFileWatcher() {
  if (fileWatcher) {
    fileWatcher.close();
  }
  
  fileWatcher = chokidar.watch([SESSION_OUTPUTS_DIR, KNOWLEDGE_OBJECTS_DIR], {
    ignored: /(^|[\/\\])\../,
    persistent: true,
    ignoreInitial: true
  });

  fileWatcher
    .on('add', path => {
      console.log(`📁 File added: ${path}`);
      io.emit('fileAdded', { path, timestamp: new Date().toISOString() });
    })
    .on('change', path => {
      console.log(`📝 File changed: ${path}`);
      io.emit('fileChanged', { path, timestamp: new Date().toISOString() });
    })
    .on('unlink', path => {
      console.log(`🗑️ File removed: ${path}`);
      io.emit('fileRemoved', { path, timestamp: new Date().toISOString() });
    });
}

// Database functions
function getSessionsFromDatabase() {
  return new Promise((resolve, reject) => {
    const dbPath = path.join(SAMPLE_DATA_DIR, 'sessions.db');
    
    if (!fs.existsSync(dbPath)) {
      console.log('📁 Sample database not found, using fallback');
      resolve([]);
      return;
    }
    
    const db = new sqlite3.Database(dbPath);
    
    const query = `
      SELECT 
        s.*,
        json_group_array(
          json_object(
            'id', f.id,
            'name', f.name,
            'path', f.path,
            'type', f.type,
            'size', f.size,
            'created', f.created,
            'modified', f.modified,
            'metadata', f.metadata
          )
        ) as files
      FROM sessions s
      LEFT JOIN files f ON s.id = f.session_id
      GROUP BY s.id
    `;
    
    db.all(query, [], (err, rows) => {
      if (err) {
        reject(err);
        return;
      }
      
      // Parse the files JSON for each session
      const sessions = rows.map(row => {
        try {
          const files = JSON.parse(row.files[0] === 'null' ? '[]' : row.files);
          return {
            ...row,
            files: files.filter(f => f.id !== null),
            startTime: row.start_time,
            endTime: row.end_time,
            fileCount: row.file_count,
            totalSize: row.total_size
          };
        } catch (e) {
          return {
            ...row,
            files: [],
            startTime: row.start_time,
            endTime: row.end_time,
            fileCount: row.file_count,
            totalSize: row.total_size
          };
        }
      });
      
      db.close();
      resolve(sessions);
    });
  });
}

// API Routes

// Get session outputs directory structure
app.get('/api/sessions', async (req, res) => {
  try {
    // Try to get sessions from sample database first
    let sessions = await getSessionsFromDatabase();
    
    // If no sample data, fall back to directory scanning
    if (!sessions || sessions.length === 0) {
      sessions = await getSessionsStructure();
    }
    
    res.json({ success: true, sessions });
  } catch (error) {
    console.error('Error getting sessions:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get specific session outputs
app.get('/api/sessions/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const sessionData = await getSessionData(sessionId);
    res.json({ success: true, session: sessionData });
  } catch (error) {
    console.error('Error getting session data:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get file content
app.get('/api/files/:filePath(*)', async (req, res) => {
  try {
    const { filePath } = req.params;
    const fullPath = path.join(__dirname, '..', filePath);
    
    if (!fs.existsSync(fullPath)) {
      return res.status(404).json({ success: false, error: 'File not found' });
    }
    
    const stats = await fs.stat(fullPath);
    const ext = path.extname(fullPath).toLowerCase();
    
    // Handle different file types
    if (ext === '.json') {
      const content = await fs.readFile(fullPath, 'utf8');
      res.json({ success: true, content: JSON.parse(content), stats });
    } else if (ext === '.csv') {
      const content = await fs.readFile(fullPath, 'utf8');
      res.json({ success: true, content, stats, type: 'csv' });
    } else if (ext === '.png' || ext === '.jpg' || ext === '.jpeg' || ext === '.svg') {
      res.sendFile(fullPath);
    } else if (ext === '.md') {
      const content = await fs.readFile(fullPath, 'utf8');
      res.json({ success: true, content, stats, type: 'markdown' });
    } else {
      // Binary or unknown file type
      res.json({ success: true, stats, type: 'binary' });
    }
  } catch (error) {
    console.error('Error reading file:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Download file
app.get('/api/download/:filePath(*)', async (req, res) => {
  try {
    const { filePath } = req.params;
    const fullPath = path.join(__dirname, '..', filePath);
    
    if (!fs.existsSync(fullPath)) {
      return res.status(404).json({ success: false, error: 'File not found' });
    }
    
    res.download(fullPath);
  } catch (error) {
    console.error('Error downloading file:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get system status
app.get('/api/status', async (req, res) => {
  try {
    const status = {
      server: 'running',
      timestamp: new Date().toISOString(),
      sessionsCount: await getSessionsCount(),
      totalFiles: await getTotalFilesCount(),
      diskUsage: await getDiskUsage()
    };
    res.json({ success: true, status });
  } catch (error) {
    console.error('Error getting status:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Helper functions
async function getSessionsStructure() {
  try {
    if (!fs.existsSync(SESSION_OUTPUTS_DIR)) {
      return [];
    }
    
    const sessions = [];
    const sessionDirs = await fs.readdir(SESSION_OUTPUTS_DIR);
    
    for (const sessionDir of sessionDirs) {
      const sessionPath = path.join(SESSION_OUTPUTS_DIR, sessionDir);
      const stats = await fs.stat(sessionPath);
      
      if (stats.isDirectory()) {
        const sessionData = await getSessionDirectoryData(sessionPath, sessionDir);
        sessions.push(sessionData);
      }
    }
    
    return sessions.sort((a, b) => new Date(b.lastModified) - new Date(a.lastModified));
  } catch (error) {
    console.error('Error reading sessions directory:', error);
    return [];
  }
}

async function getSessionDirectoryData(sessionPath, sessionDir) {
  try {
    const files = await fs.readdir(sessionPath);
    const fileStats = [];
    let totalSize = 0;
    
    for (const file of files) {
      const filePath = path.join(sessionPath, file);
      const stats = await fs.stat(filePath);
      totalSize += stats.size;
      
      fileStats.push({
        name: file,
        path: path.join(sessionPath, file),
        size: stats.size,
        sizeFormatted: formatFileSize(stats.size),
        type: getFileType(file),
        lastModified: stats.mtime,
        isDirectory: stats.isDirectory()
      });
    }
    
    return {
      id: sessionDir,
      name: sessionDir,
      path: sessionPath,
      files: fileStats,
      totalSize,
      totalSizeFormatted: formatFileSize(totalSize),
      lastModified: Math.max(...fileStats.map(f => f.lastModified.getTime())),
      fileCount: fileStats.length
    };
  } catch (error) {
    console.error('Error reading session directory:', error);
    return { id: sessionDir, name: sessionDir, error: error.message };
  }
}

async function getSessionData(sessionId) {
  const sessionPath = path.join(SESSION_OUTPUTS_DIR, sessionId);
  if (!fs.existsSync(sessionPath)) {
    throw new Error('Session not found');
  }
  
  return await getSessionDirectoryData(sessionPath, sessionId);
}

async function getSessionsCount() {
  try {
    if (!fs.existsSync(SESSION_OUTPUTS_DIR)) return 0;
    const items = await fs.readdir(SESSION_OUTPUTS_DIR);
    const dirs = [];
    for (const item of items) {
      const itemPath = path.join(SESSION_OUTPUTS_DIR, item);
      const stats = await fs.stat(itemPath);
      if (stats.isDirectory()) dirs.push(item);
    }
    return dirs.length;
  } catch (error) {
    return 0;
  }
}

async function getTotalFilesCount() {
  try {
    let count = 0;
    if (fs.existsSync(SESSION_OUTPUTS_DIR)) {
      const walk = async (dir) => {
        const items = await fs.readdir(dir);
        for (const item of items) {
          const itemPath = path.join(dir, item);
          const stats = await fs.stat(itemPath);
          if (stats.isDirectory()) {
            await walk(itemPath);
          } else {
            count++;
          }
        }
      };
      await walk(SESSION_OUTPUTS_DIR);
    }
    return count;
  } catch (error) {
    return 0;
  }
}

async function getDiskUsage() {
  try {
    const stats = await fs.stat(SESSION_OUTPUTS_DIR);
    return {
      total: stats.size,
      totalFormatted: formatFileSize(stats.size)
    };
  } catch (error) {
    return { total: 0, totalFormatted: '0 B' };
  }
}

function formatFileSize(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function getFileType(filename) {
  const ext = path.extname(filename).toLowerCase();
  const imageExts = ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.bmp'];
  const docExts = ['.pdf', '.doc', '.docx', '.txt', '.md'];
  const dataExts = ['.csv', '.json', '.xml', '.yaml', '.yml'];
  const archiveExts = ['.zip', '.tar', '.gz', '.rar', '.7z'];
  
  if (imageExts.includes(ext)) return 'image';
  if (docExts.includes(ext)) return 'document';
  if (dataExts.includes(ext)) return 'data';
  if (archiveExts.includes(ext)) return 'archive';
  if (ext === '.db') return 'database';
  if (ext === '.log') return 'log';
  return 'unknown';
}

// Socket.IO connection handling
io.on('connection', (socket) => {
  console.log(`🔌 Client connected: ${socket.id}`);
  
  socket.on('disconnect', () => {
    console.log(`🔌 Client disconnected: ${socket.id}`);
  });
  
  socket.on('requestSessions', async () => {
    try {
      const sessions = await getSessionsStructure();
      socket.emit('sessionsUpdate', sessions);
    } catch (error) {
      socket.emit('error', { message: error.message });
    }
  });
});

// Serve React app for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'client', 'build', 'index.html'));
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down session viewer server...');
  if (fileWatcher) {
    fileWatcher.close();
  }
  server.close(() => {
    console.log('✅ Server closed gracefully');
    process.exit(0);
  });
});

// Start server
server.listen(PORT, () => {
  console.log(`🚀 Cybersecurity Agent Session Viewer running on port ${PORT}`);
  console.log(`📁 Monitoring: ${SESSION_OUTPUTS_DIR}`);
  console.log(`🔗 Open: http://localhost:${PORT}`);
  
  // Initialize file watcher
  initializeFileWatcher();
});

module.exports = { app, server, io };
