# Session Viewer Usage Guide

## Overview

The **Auto-Managing Session Viewer** is a professional web interface that automatically launches when needed during complex cybersecurity workflows and shuts down when not in use. It provides an interactive way to browse, analyze, and interact with workflow outputs without any manual management.

## Key Features

### 🚀 **Auto-Management**
- **Automatic Launch**: Starts when requested by workflows or user commands
- **Auto-Shutdown**: Closes after 5 minutes of inactivity to reclaim resources
- **Smart Lifecycle**: Manages dependencies, builds, and server processes automatically

### 🎨 **Professional Interface**
- **Cybersecurity Theme**: Modern, dark interface with cyber-accent colors
- **Real-time Updates**: Live file system monitoring with WebSocket updates
- **Interactive Elements**: Zoom, pan, download, and preview capabilities
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### 🔧 **Seamless Integration**
- **MCP Tools**: Available as workflow tools for the Planner Agent
- **CLI Integration**: Clickable links and status information in terminal
- **Resource Management**: Automatic cleanup and resource reclamation

## How It Works

### 1. **Automatic Launch**
When a workflow needs to show outputs or a user requests the viewer:

```python
# In workflow planning
workflow_steps.append({
    'action': 'launch_session_viewer',
    'description': 'Show interactive output browser',
    'reason': 'Complex workflow with multiple output files'
})
```

### 2. **Smart Resource Management**
- **Dependencies**: Automatically installs Node.js packages if needed
- **Build Process**: Compiles React client on first use
- **Server Management**: Starts/stops Node.js server as needed
- **Port Management**: Uses port 3001 by default (configurable)

### 3. **Auto-Shutdown**
- **Inactivity Timer**: 5 minutes of no activity triggers shutdown
- **Resource Cleanup**: Properly terminates processes and reclaims memory
- **Graceful Shutdown**: Sends SIGTERM, waits for cleanup, then SIGKILL if needed

## Usage Scenarios

### **Scenario 1: Complex Workflow Outputs**
```
User: "Analyze this network traffic and show me the results visually"

Planner Agent: "I'll analyze the traffic and launch the session viewer to show you the outputs."

Workflow:
1. Run PCAP analysis
2. Generate visualizations
3. Launch session viewer ← Auto-launches here
4. Present results in browser
5. Continue with CLI chat
```

### **Scenario 2: User Request**
```
User: "Can you show me the session viewer for my last workflow?"

Agent: "I'll launch the session viewer to show you the outputs from your last workflow."

Result: Browser opens with session viewer, user can browse files
```

### **Scenario 3: Resource Management**
```
After 5 minutes of inactivity:
⏰ Auto-shutdown after 300 seconds of inactivity
🛑 Stopping session viewer...
✅ Session viewer stopped successfully
💡 Resources reclaimed successfully
```

## MCP Tools Available

### **1. launch_session_viewer**
Launches the session viewer with optional browser opening.

**Parameters:**
- `auto_open_browser` (boolean): Whether to open browser automatically (default: true)
- `reason` (string): Reason for launching (for logging)

**Example:**
```python
result = tools.launch_session_viewer(
    auto_open_browser=True,
    reason="Complex workflow with multiple visualization outputs"
)
```

### **2. get_session_viewer_status**
Gets current viewer status and provides CLI-friendly output.

**Example:**
```python
status = tools.get_session_viewer_status()
# Returns status with formatted CLI output
```

### **3. extend_session_viewer**
Extends the current session by resetting the auto-shutdown timer.

**Parameters:**
- `reason` (string): Reason for extending the session

**Example:**
```python
tools.extend_session_viewer(reason="User still analyzing outputs")
```

### **4. stop_session_viewer**
Manually stops the viewer and reclaims resources.

**Parameters:**
- `reason` (string): Reason for stopping

**Example:**
```python
tools.stop_session_viewer(reason="Workflow completed, user finished")
```

### **5. open_session_viewer_url**
Opens the viewer URL in the default browser.

**Example:**
```python
tools.open_session_viewer_url()
```

## CLI Output Examples

### **Launch Success:**
```
🔗 **Session Viewer Launched!**
   **URL:** http://localhost:3001
   **Status:** started
   **Auto-shutdown:** 5 minutes

💡 **Usage:**
   • Click the URL above to open in browser
   • Browse session outputs and files
   • Viewer will auto-close after 5 minutes of inactivity
   • Use 'extend_session_viewer' to keep it open longer
```

### **Status Information:**
```
🟢 **Session Viewer Status:**
   **Status:** Running
   **URL:** http://localhost:3001
   **Uptime:** 0:02:15
   **Auto-shutdown:** 2m 45s

💡 **Actions:**
   • Click URL to open in browser
   • Use 'extend_session_viewer' to keep open
   • Use 'stop_session_viewer' to close now
```

### **Stop Confirmation:**
```
🛑 **Session Viewer Stopped**
   **Status:** stopped
   **Message:** Session viewer stopped successfully
   **Reason:** Workflow completed

💡 **Resources reclaimed successfully**
```

## Technical Details

### **File Structure:**
```
session-viewer/
├── server.js              # Express server with file monitoring
├── client/                # React application
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── context/       # React context providers
│   │   └── index.js       # Main entry point
│   ├── public/            # Static assets
│   └── package.json       # React dependencies
├── package.json           # Server dependencies
└── README.md             # Setup instructions
```

### **Dependencies:**
- **Server**: Express, Socket.IO, Chokidar, Helmet
- **Client**: React 18, Tailwind CSS, Framer Motion, Lucide React
- **System**: Node.js 16+, npm

### **Ports & Security:**
- **Default Port**: 3001 (configurable via environment)
- **Access**: Localhost only (no external access)
- **Security**: Helmet.js headers, CORS configuration
- **File Access**: Restricted to project directories

## Troubleshooting

### **Common Issues:**

#### **1. Node.js Not Available**
```
❌ Node.js not available - session viewer will not be available
```
**Solution**: Install Node.js 16+ from [nodejs.org](https://nodejs.org/)

#### **2. Dependencies Installation Failed**
```
❌ Failed to install dependencies: npm error
```
**Solution**: 
```bash
cd session-viewer
rm -rf node_modules package-lock.json
npm install
```

#### **3. Build Failed**
```
❌ Failed to build client: build error
```
**Solution**:
```bash
cd session-viewer/client
npm install
cd ..
npm run build-client
```

#### **4. Port Already in Use**
```
❌ Server failed to start: port already in use
```
**Solution**: Change port in `server.js` or kill existing process

### **Debug Commands:**
```bash
# Test the manager directly
python bin/session_viewer_manager.py

# Test MCP tools
python bin/session_viewer_mcp_tools.py

# Test full integration
python test_session_viewer_integration.py

# Check Node.js
node --version
npm --version

# Check session viewer directory
ls -la session-viewer/
```

## Best Practices

### **1. Workflow Integration**
- Launch viewer when you have multiple output files to show
- Use descriptive reasons for logging and debugging
- Consider user workflow context when deciding to launch

### **2. Resource Management**
- Let auto-shutdown handle cleanup when possible
- Use `extend_session_viewer` if user needs more time
- Manually stop if workflow is complete and user is done

### **3. User Experience**
- Provide clear instructions in CLI output
- Use clickable links when possible
- Explain what the viewer shows and how to use it

### **4. Error Handling**
- Always check return values from MCP tools
- Provide helpful error messages to users
- Log issues for debugging

## Future Enhancements

### **Planned Features:**
- **Custom Ports**: User-configurable port selection
- **Session Persistence**: Remember user preferences across sessions
- **Advanced Filtering**: File type, date range, and search filters
- **Export Capabilities**: Bulk download and export features
- **Integration APIs**: REST APIs for external tool integration

### **Customization Options:**
- **Theme Selection**: Multiple cybersecurity themes
- **Layout Options**: Customizable dashboard layouts
- **Notification Settings**: Configurable alerts and updates
- **Access Control**: Optional authentication for team environments

## Support & Development

### **Getting Help:**
1. Check the troubleshooting section above
2. Run the test scripts to verify setup
3. Check console logs for detailed error information
4. Review the Node.js and React documentation

### **Contributing:**
1. Follow the existing code style and patterns
2. Add tests for new features
3. Update documentation for changes
4. Ensure responsive design works on all devices

### **Reporting Issues:**
Include the following information:
- Error messages and stack traces
- System information (OS, Node.js version)
- Steps to reproduce the issue
- Expected vs. actual behavior

---

**The Session Viewer is designed to be invisible to users - it just works when needed and cleans up after itself. Focus on your cybersecurity workflows, and let the viewer handle the presentation details automatically!**
