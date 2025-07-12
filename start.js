// TrialMatch/start.js
const path = require('path');
const { spawn } = require('child_process');

const nodeAppPath = path.join(__dirname, 'Node', 'app.js'); // Assuming Node.js entry point is app.js
const pythonScriptPath = path.join(__dirname, 'BE', 'app.py');

// Example: Pass the pythonScriptPath as an environment variable or command-line argument
// to your Node.js application. For simplicity, let's use an environment variable here.

console.log('Starting Node.js application...');
const nodeProcess = spawn('node', [nodeAppPath], {
    env: { ...process.env, PYTHON_SCRIPT_ABS_PATH: pythonScriptPath },
    cwd: path.dirname(nodeAppPath), // Set CWD for the Node app
    stdio: 'inherit' // Pipe output to parent process
});

nodeProcess.on('close', (code) => {
    console.log(`Node.js application exited with code ${code}`);
});

nodeProcess.on('error', (err) => {
    console.error('Failed to start Node.js application:', err);
});