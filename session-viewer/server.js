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
// Make sqlite3 optional - use fallback if not available
let sqlite3 = null;
try {
  sqlite3 = require('sqlite3').verbose();
} catch (error) {
  console.log('⚠️  sqlite3 not available, using file-based fallback');
}

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
      styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
      fontSrc: ["'self'", "https://fonts.gstatic.com"],
      scriptSrc: ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
      imgSrc: ["'self'", "data:", "blob:"],
      connectSrc: ["'self'", "ws:", "wss:", "http://localhost:3001"],
      workerSrc: ["'self'", "blob:"]
    }
  }
}));
app.use(compression());
app.use(cors());
app.use(morgan('combined'));
app.use(express.json({ limit: '50mb' }));
// Serve React app from build directory first (takes priority)
app.use(express.static(path.join(__dirname, 'client', 'build')));

// Serve static files from public directory (fallback for development resources)
app.use(express.static(path.join(__dirname, 'client', 'public')));

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
    // If sqlite3 is not available, use file-based fallback
    if (!sqlite3) {
      console.log('📁 sqlite3 not available, using file-based fallback');
      resolve([]);
      return;
    }
    
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

// API routes are now defined after the Socket.IO section to ensure proper order

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
    const fileStats = [];
    let totalSize = 0;
    
    // Recursively scan for all files in the session directory
    const walk = async (dir, relativePath = '') => {
      const items = await fs.readdir(dir);
      
      for (const item of items) {
        const itemPath = path.join(dir, item);
        const stats = await fs.stat(itemPath);
        const relativeItemPath = relativePath ? path.join(relativePath, item) : item;
        
        if (stats.isDirectory()) {
          // Recursively scan subdirectories
          await walk(itemPath, relativeItemPath);
        } else {
          // Add file to the list
          totalSize += stats.size;
          fileStats.push({
            name: item,
            path: relativeItemPath, // Use relative path for display
            fullPath: itemPath,     // Keep full path for operations
            size: stats.size,
            sizeFormatted: formatFileSize(stats.size),
            type: getFileType(item),
            lastModified: stats.mtime,
            isDirectory: false
          });
        }
      }
    };
    
    // Start recursive scan from the session directory
    await walk(sessionPath);
    
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

// API routes must come BEFORE the catch-all route
app.get('/api/sessions', async (req, res) => {
  try {
    const sessions = await getSessionsStructure();
    res.json({ sessions });
  } catch (error) {
    console.error('Error fetching sessions:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/sessions/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const sessions = await getSessionsStructure();
    const session = sessions.find(s => s.id === sessionId);
    
    if (!session) {
      return res.status(404).json({ error: 'Session not found' });
    }
    
    res.json({ success: true, session });
  } catch (error) {
    console.error('Error fetching session:', error);
    res.status(500).json({ error: error.message });
  }
});

// Serve raw file content for viewing
app.get('/api/view/:filePath(*)', async (req, res) => {
  try {
    const { filePath } = req.params;
    // URL decode the file path to handle encoded characters like %2F
    const decodedFilePath = decodeURIComponent(filePath);
    const fullPath = path.join(SESSION_OUTPUTS_DIR, decodedFilePath);
    
    if (!fs.existsSync(fullPath)) {
      return res.status(404).send('File not found');
    }
    
    const stats = await fs.stat(fullPath);
    if (stats.isDirectory()) {
      return res.status(400).send('Cannot read directory');
    }
    
    // Determine content type based on file extension
    const ext = path.extname(fullPath).toLowerCase();
    let contentType = 'text/plain';
    
    if (ext === '.html' || ext === '.htm') {
      contentType = 'text/html';
    } else if (ext === '.css') {
      contentType = 'text/css';
    } else if (ext === '.js') {
      contentType = 'application/javascript';
    } else if (ext === '.json') {
      contentType = 'application/json';
    } else if (ext === '.xml') {
      contentType = 'application/xml';
    } else if (ext === '.csv') {
      contentType = 'text/csv';
    } else if (ext === '.pdf') {
      contentType = 'application/pdf';
    } else if (ext.match(/\.(png|jpg|jpeg|gif|svg)$/)) {
      contentType = `image/${ext.slice(1)}`;
    }
    
    res.setHeader('Content-Type', contentType);
    res.setHeader('Content-Disposition', `inline; filename="${path.basename(fullPath)}"`);
    
    // For binary files, serve as binary
    if (ext === '.pdf' || ext.match(/\.(png|jpg|jpeg|gif|svg)$/)) {
      const content = await fs.readFile(fullPath);
      res.send(content);
    } else {
      // For text files, serve as text
      const content = await fs.readFile(fullPath, 'utf8');
      res.send(content);
    }
  } catch (error) {
    console.error('Error serving file:', error);
    res.status(500).send('Error serving file');
  }
});

// API endpoint for programmatic access (returns JSON)
app.get('/api/files/:filePath(*)', async (req, res) => {
  try {
    const { filePath } = req.params;
    // URL decode the file path to handle encoded characters like %2F
    const decodedFilePath = decodeURIComponent(filePath);
    const fullPath = path.join(SESSION_OUTPUTS_DIR, decodedFilePath);
    
    if (!fs.existsSync(fullPath)) {
      return res.status(404).json({ error: 'File not found' });
    }
    
    const stats = await fs.stat(fullPath);
    if (stats.isDirectory()) {
      return res.status(400).json({ error: 'Cannot read directory' });
    }
    
    const content = await fs.readFile(fullPath, 'utf8');
    res.json({ content, path: filePath });
  } catch (error) {
    console.error('Error reading file:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/download/:filePath(*)', async (req, res) => {
  try {
    const { filePath } = req.params;
    // URL decode the file path to handle encoded characters like %2F
    const decodedFilePath = decodeURIComponent(filePath);
    const fullPath = path.join(SESSION_OUTPUTS_DIR, decodedFilePath);
    
    if (!fs.existsSync(fullPath)) {
      return res.status(404).json({ error: 'File not found' });
    }
    
    const stats = await fs.stat(fullPath);
    if (stats.isDirectory()) {
      return res.status(400).json({ error: 'Cannot download directory' });
    }
    
    res.download(fullPath);
  } catch (error) {
    console.error('Error downloading file:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/status', async (req, res) => {
  try {
    const sessionCount = await getSessionsCount();
    const diskUsage = await getDiskUsage();
    
    res.json({
      status: {
        server: 'running',
        timestamp: new Date().toISOString(),
        sessionCount,
        diskUsage
      }
    });
  } catch (error) {
    console.error('Error getting status:', error);
    res.status(500).json({ error: error.message });
  }
});

// Serve React app for all other routes (must be LAST)
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
