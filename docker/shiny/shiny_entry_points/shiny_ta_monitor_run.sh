#!/bin/sh

until cd /shiny_apps
do
    echo "Waiting for server volume..."
done

shiny run target_acq_monitor.py --host 0.0.0.0 --port 8000 --log-level debug

tail -f /dev/null
