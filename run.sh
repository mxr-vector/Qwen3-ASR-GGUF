#!/bin/bash

# Configuration
APP_MODULE="infer:app"
HOST="0.0.0.0"
PORT=8002
PID_FILE="logs/app.pid"

# ─── Docker 前台模式（推荐） ──────────────────────────────────────────────
# 在 Docker 容器中必须使用此模式：uvicorn 作为 PID 1 前台运行，
# 确保信号能正确传递（SIGTERM 优雅关闭）、日志输出到 stdout、
# 长连接 SSE 流不会因容器进程管理机制而被意外中断。
serve() {
    echo "Starting Qwen3-ASR Fast API Service (foreground mode)..."
    mkdir -p logs
    exec uvicorn "$APP_MODULE" \
        --host "$HOST" \
        --port "$PORT" \
        --timeout-keep-alive 600
}

# ─── 后台模式（仅限裸机/VM 部署） ───────────────────────────────────────
start() {
    echo "Starting Qwen3-ASR Fast API Service..."
    if [ -f "$PID_FILE" ]; then
        if kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            echo "Service is already running with PID $(cat $PID_FILE)."
            exit 1
        else
            echo "Removing stale PID file $PID_FILE"
            rm -f "$PID_FILE"
        fi
    fi
    
    # 确保 logs 目录存在
    mkdir -p logs

    # 这里使用 nohup 将 uvicorn 放入后台启动
    nohup uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT" --timeout-keep-alive 600 > logs/app.log 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    echo "Service started with PID $PID in the background."
}

stop() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "Stopping Qwen3-ASR Fast API Service with PID $PID..."
        kill -15 "$PID" 2>/dev/null
        rm -f "$PID_FILE"
        echo "Service stopped."
    else
        echo "Service is not running (PID file not found)."
    fi
}

restart() {
    stop
    sleep 2
    start
}

case "$1" in
    serve)
        serve
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    *)
        echo "Usage: $0 {serve|start|stop|restart}"
        echo "  serve   - 前台运行 (Docker 容器必须用此模式)"
        echo "  start   - 后台运行 (裸机/VM 部署)"
        echo "  stop    - 停止后台服务"
        echo "  restart - 重启后台服务"
        exit 1
esac
