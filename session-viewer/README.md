# Cybersecurity Agent Session Viewer

A professional, real-time web interface for viewing and managing cybersecurity workflow outputs.

## Features

- 🎨 **Professional Cybersecurity Theme** - Modern, dark interface with cyber-accent colors
- 📁 **Real-time File Monitoring** - Live updates when new session outputs are created
- 🔍 **Interactive File Viewing** - Support for images, documents, data files, and more
- 📊 **Dashboard Analytics** - Overview of sessions, file types, and system status
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile devices
- ⚡ **Fast Performance** - Built with React and optimized for large file collections
- 🔐 **Secure Access** - Local server with proper security headers

## Quick Start

### Prerequisites

- Node.js 16+ 
- npm or yarn

### Installation

1. **Install server dependencies:**
   ```bash
   cd session-viewer
   npm install
   ```

2. **Install client dependencies:**
   ```bash
   cd client
   npm install
   cd ..
   ```

3. **Build the React client:**
   ```bash
   npm run build-client
   ```

4. **Start the server:**
   ```bash
   npm start
   ```

5. **Open your browser:**
   Navigate to `http://localhost:3001`

## Development

### Development Mode

For development with hot reloading:

```bash
# Terminal 1 - Start the server
npm run dev

# Terminal 2 - Start the React dev server
cd client
npm start
```

### Building for Production

```bash
npm run build
```

## Architecture

### Server (Node.js + Express)
- **File Monitoring** - Uses `chokidar` for real-time file system watching
- **API Endpoints** - RESTful APIs for session and file management
- **WebSocket Support** - Real-time updates via Socket.IO
- **Security** - Helmet.js for security headers, CORS configuration

### Client (React)
- **Modern UI** - Built with React 18, Tailwind CSS, and Framer Motion
- **State Management** - React Query for server state, Context for app state
- **Real-time Updates** - Socket.IO client for live file system changes
- **Responsive Design** - Mobile-first approach with Tailwind breakpoints

## API Endpoints

- `GET /api/sessions` - List all sessions
- `GET /api/sessions/:id` - Get specific session details
- `GET /api/files/:path` - Get file content
- `GET /api/download/:path` - Download file
- `GET /api/status` - System status

## File Types Supported

- **Images** - PNG, JPG, JPEG, SVG, GIF, BMP
- **Documents** - PDF, DOC, DOCX, TXT, MD
- **Data Files** - CSV, JSON, XML, YAML
- **Databases** - SQLite files
- **Archives** - ZIP, TAR, GZ, RAR, 7Z
- **Logs** - Log files
- **Unknown** - Binary and other file types

## Integration with Cybersecurity Agent

This viewer is designed to be launched by the Planner Agent during complex workflows:

1. **Workflow Planning** - Agent can add "show session viewer" to workflow
2. **Real-time Monitoring** - View outputs as they're generated
3. **Interactive Analysis** - Zoom, pan, and interact with visualizations
4. **Seamless Return** - Close browser tab to return to CLI chat

## Customization

### Colors and Theme
Edit `client/tailwind.config.js` to customize the cybersecurity color scheme.

### File Type Icons
Modify the icon mappings in `client/src/components/Dashboard.js`.

### API Configuration
Update server endpoints in `server.js` for custom integrations.

## Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Change port in server.js
   const PORT = process.env.PORT || 3002;
   ```

2. **File watching not working:**
   - Ensure the `session-outputs` directory exists
   - Check file permissions

3. **Build errors:**
   ```bash
   # Clean and reinstall
   rm -rf node_modules package-lock.json
   npm install
   ```

### Logs

Check server logs for detailed error information:
```bash
npm run dev
```

## Security Notes

- Server runs locally only (localhost)
- No authentication required for local development
- File access restricted to project directories
- HTTPS recommended for production use

## Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure responsive design works

## License

MIT License - see LICENSE file for details.
