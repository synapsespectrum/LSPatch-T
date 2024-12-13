#!/bin/sh

# MLFlow server configuration
BENCHMARK_HOST="127.0.0.1"
BENCHMARK_PORT="8080"
BENCHMARK_BACKEND_URI="file:./mlruns"

BASELINE_HOST="127.0.0.1"
BASELINE_PORT="5000"
BASELINE_BACKEND_URI="file:./mlrun_prototype"

MLFLOW_APP_NAME="basic-auth"
MLFLOW_USER="andrew"
MLFLOW_PASS="a"

# Function to set the correct configuration based on the service type
set_config() {
    case "$1" in
        benchmark)
            MLFLOW_HOST=$BENCHMARK_HOST
            MLFLOW_PORT=$BENCHMARK_PORT
            MLFLOW_BACKEND_URI=$BENCHMARK_BACKEND_URI
            ;;
        baseline)
            MLFLOW_HOST=$BASELINE_HOST
            MLFLOW_PORT=$BASELINE_PORT
            MLFLOW_BACKEND_URI=$BASELINE_BACKEND_URI
            ;;
        *)
            echo "Invalid service type. Use 'benchmark' or 'baseline'."
            exit 1
            ;;
    esac
}

check_mlflow_running() {
    sudo lsof -i -P -n | grep LISTEN | grep :$MLFLOW_PORT > /dev/null
    return $?
}

get_mlflow_pid() {
    sudo lsof -i -P -n | grep LISTEN | grep :$MLFLOW_PORT | awk '{print $2}' | uniq
}

start_mlflow() {
    if check_mlflow_running; then
        echo "MLFlow server ($1) is already running on port $MLFLOW_PORT"
        return
    fi
    echo "Starting MLFlow server ($1)..."
    nohup mlflow server --backend-store-uri $MLFLOW_BACKEND_URI --host $MLFLOW_HOST --port $MLFLOW_PORT --app-name $MLFLOW_APP_NAME > /dev/null 2>&1 &
    sleep 2
    if check_mlflow_running; then
        echo "MLFlow server ($1) started successfully"
        echo "Username: $MLFLOW_USER"
        echo "Password: $MLFLOW_PASS"
    else
        echo "Failed to start MLFlow server ($1)"
    fi
}

stop_mlflow() {
    if ! check_mlflow_running; then
        echo "MLFlow server ($1) is not running on port $MLFLOW_PORT"
        return
    fi
    echo "Stopping MLFlow server ($1)..."
    PID=$(get_mlflow_pid)
    if [ -n "$PID" ]; then
        sudo kill $PID
        sleep 2
        if check_mlflow_running; then
            echo "MLFlow server ($1) did not stop gracefully. Forcing stop..."
            sudo kill -9 $PID
        fi
        echo "MLFlow server ($1) stopped"
    else
        echo "Could not find PID for MLFlow server ($1)"
    fi
}

status_mlflow() {
    if check_mlflow_running; then
        PID=$(get_mlflow_pid)
        echo "MLFlow server ($1) is running on port $MLFLOW_PORT"
        echo "Process ID: $PID"
        echo "Host: $MLFLOW_HOST"
        echo "Port: $MLFLOW_PORT"
        echo "App Name: $MLFLOW_APP_NAME"
        echo "Username: $MLFLOW_USER"
        echo "Password: $MLFLOW_PASS"
        echo "Backend Store URI: $MLFLOW_BACKEND_URI"
    else
        echo "MLFlow server ($1) is not running on port $MLFLOW_PORT"
    fi
}

# Check if correct number of arguments is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 {benchmark|baseline} {start|stop|restart|status}"
    exit 1
fi

# Set the configuration based on the first argument
set_config $1

# Execute the command based on the second argument
case "$2" in
    start)
        start_mlflow $1
        ;;
    stop)
        stop_mlflow $1
        ;;
    restart)
        stop_mlflow $1
        start_mlflow $1
        ;;
    status)
        status_mlflow $1
        ;;
    *)
        echo "Usage: $0 {benchmark|baseline} {start|stop|restart|status}"
        exit 1
        ;;
esac

exit 0