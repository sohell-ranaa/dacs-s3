#!/bin/bash

# Jupyter Lab Start/Stop Script
# DACS - Data Analytics in Cyber Security

DEFAULT_PORT=8888

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Show menu
show_menu() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}   Jupyter Notebook Manager${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    echo "1) Start Jupyter Notebook"
    echo "2) Stop Jupyter Notebook"
    echo "3) Status"
    echo "4) Exit"
    echo ""
}

# Start Jupyter
start_jupyter() {
    echo -e "${BLUE}Select root folder:${NC}"
    echo ""
    echo "1) Assignment (default)"
    echo "2) Assignment/Starter_Work"
    echo "3) Assignment/notebooks"
    echo "4) ClassMaterials"
    echo "5) DACS (root)"
    echo "6) Custom path"
    echo ""
    printf "Choice [1]: "
    read -r folder_choice

    case $folder_choice in
        1|"") FOLDER="/home/rana-workspace/DACS/Assignment" ;;
        2) FOLDER="/home/rana-workspace/DACS/Assignment/Starter_Work" ;;
        3) FOLDER="/home/rana-workspace/DACS/Assignment/notebooks" ;;
        4) FOLDER="/home/rana-workspace/DACS/ClassMaterials" ;;
        5) FOLDER="/home/rana-workspace/DACS" ;;
        6)
            printf "Enter path: "
            read -r FOLDER
            ;;
        *) FOLDER="/home/rana-workspace/DACS/Assignment" ;;
    esac

    echo ""
    printf "Port [%s]: " "$DEFAULT_PORT"
    read -r PORT
    PORT=${PORT:-$DEFAULT_PORT}

    echo ""

    # Check if already running
    if lsof -i :$PORT > /dev/null 2>&1; then
        echo -e "${YELLOW}Port $PORT is already in use${NC}"
        echo "Stop it first or choose a different port."
        return 1
    fi

    echo -e "${GREEN}Starting Jupyter Notebook...${NC}"
    echo -e "  Folder: ${YELLOW}$FOLDER${NC}"
    echo -e "  Port:   ${YELLOW}$PORT${NC}"
    echo ""

    cd "$FOLDER"

    # Start jupyter in background (using notebook, not lab)
    nohup jupyter-notebook \
        --ip=0.0.0.0 \
        --port=$PORT \
        --no-browser \
        --allow-root \
        --NotebookApp.token='' \
        --NotebookApp.password='' \
        --notebook-dir="$FOLDER" \
        > /tmp/jupyter_lab.log 2>&1 &

    echo "Waiting for Jupyter to start..."
    sleep 3

    if lsof -i :$PORT > /dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}✓ Jupyter Notebook started!${NC}"
        echo ""
        echo -e "URL: ${BLUE}http://localhost:$PORT${NC}"
        echo "Log: /tmp/jupyter_lab.log"
    else
        echo ""
        echo -e "${RED}✗ Failed to start${NC}"
        echo "Check log: cat /tmp/jupyter_lab.log"
    fi
}

# Stop Jupyter
stop_jupyter() {
    printf "Port to stop [%s]: " "$DEFAULT_PORT"
    read -r PORT
    PORT=${PORT:-$DEFAULT_PORT}

    echo ""
    echo -e "${YELLOW}Stopping Jupyter on port $PORT...${NC}"

    # 1. Stop PM2 jupyter processes
    if command -v pm2 &> /dev/null; then
        if pm2 list 2>/dev/null | grep -qi jupyter; then
            echo "Stopping PM2 Jupyter processes..."
            pm2 stop jupyter-assignment 2>/dev/null
            pm2 delete jupyter-assignment 2>/dev/null
            echo -e "${GREEN}✓ PM2 jupyter stopped${NC}"
        fi
    fi

    # 2. Kill process on port
    PIDS=$(lsof -t -i :$PORT 2>/dev/null)
    if [ -n "$PIDS" ]; then
        kill -9 $PIDS 2>/dev/null
        sleep 1
        echo -e "${GREEN}✓ Killed processes on port $PORT${NC}"
    fi

    # 3. Kill remaining jupyter
    pkill -9 -f "jupyter-lab" 2>/dev/null
    pkill -9 -f "jupyter-notebook" 2>/dev/null

    sleep 1
    if lsof -i :$PORT > /dev/null 2>&1; then
        echo -e "${RED}✗ Port $PORT still in use!${NC}"
    else
        echo -e "${GREEN}✓ Port $PORT is now free${NC}"
    fi
}

# Show status
show_status() {
    echo -e "${BLUE}Jupyter Status:${NC}"
    echo ""

    for port in 8888 8889 8890; do
        if lsof -i :$port > /dev/null 2>&1; then
            echo -e "  Port $port: ${GREEN}RUNNING${NC}"
        else
            echo -e "  Port $port: ${RED}stopped${NC}"
        fi
    done

    echo ""
    echo "PM2 Jupyter:"
    if command -v pm2 &> /dev/null; then
        pm2 list 2>/dev/null | grep -i jupyter || echo "  None"
    else
        echo "  PM2 not installed"
    fi

    echo ""
    echo "System Jupyter:"
    pgrep -fa "jupyter" 2>/dev/null | grep -v grep || echo "  None"
}

# Main
while true; do
    show_menu
    printf "Choice: "
    read -r choice
    echo ""

    case $choice in
        1) start_jupyter ;;
        2) stop_jupyter ;;
        3) show_status ;;
        4) echo "Bye!"; exit 0 ;;
        *) echo -e "${RED}Invalid choice${NC}" ;;
    esac

    echo ""
    printf "Press Enter to continue..."
    read -r
    clear
done
