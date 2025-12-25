#!/bin/bash
# Run script for Realtime Webcam 3DGS

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Realtime Webcam 3D Gaussian Splatting ===${NC}"
echo ""

# Check if Python environment is set up
if [ ! -d "../ml-sharp/ml-sharp" ]; then
    echo -e "${RED}Error: ml-sharp not found at ../ml-sharp/ml-sharp${NC}"
    echo "Please ensure ml-sharp is properly cloned."
    exit 1
fi

# Check for virtual environment
VENV_PATH=""
if [ -d "../venv" ]; then
    VENV_PATH="../venv"
elif [ -d "venv" ]; then
    VENV_PATH="venv"
fi

if [ -n "$VENV_PATH" ]; then
    echo -e "${GREEN}Found virtual environment at $VENV_PATH${NC}"
    PYTHON="$VENV_PATH/bin/python3"
else
    echo -e "${YELLOW}No virtual environment found. Using system Python.${NC}"
    PYTHON="python3"
fi

# Check Python dependencies
echo "Checking Python dependencies..."
if ! $PYTHON -c "import torch; import sharp" 2>/dev/null; then
    echo -e "${YELLOW}Warning: Python dependencies may not be installed.${NC}"
    echo "Please run:"
    echo "  python3 -m venv ../venv"
    echo "  source ../venv/bin/activate"
    echo "  cd ../ml-sharp/ml-sharp && pip install -r requirements.txt && pip install -e ."
    echo ""
fi

# Create temp directories
mkdir -p /tmp/webcam_3dgs/captures
mkdir -p /tmp/webcam_3dgs/outputs

# Function to cleanup
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    # Kill any running SHARP server
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
    fi
    # Remove socket file
    rm -f /tmp/webcam_3dgs/server.sock
    echo -e "${GREEN}Done.${NC}"
}
trap cleanup EXIT

# Start SHARP server in background
echo ""
echo -e "${GREEN}Starting SHARP server...${NC}"
echo "This may take 10-20 seconds for initial model download."
echo ""

$PYTHON sharp_server.py --device mps &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to initialize..."
for i in {1..60}; do
    if [ -S /tmp/webcam_3dgs/server.sock ]; then
        # Try to connect
        if $PYTHON -c "
import socket
import json
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    s.connect('/tmp/webcam_3dgs/server.sock')
    s.sendall(json.dumps({'command': 'ping'}).encode())
    s.shutdown(socket.SHUT_WR)
    response = s.recv(1024)
    exit(0)
except:
    exit(1)
finally:
    s.close()
" 2>/dev/null; then
            echo -e "${GREEN}SHARP server is ready!${NC}"
            break
        fi
    fi
    sleep 1
    echo -n "."
done
echo ""

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}Error: SHARP server failed to start.${NC}"
    echo "Check if all dependencies are installed."
    exit 1
fi

# Build and run the Swift app
echo ""
echo -e "${GREEN}Building Swift application...${NC}"
swift build

echo ""
echo -e "${GREEN}Starting application...${NC}"
echo "Press Ctrl+C to quit."
echo ""

swift run RealtimeWebcam3DGS
