#!/bin/bash
source venv/bin/activate
#exec gunicorn -b :5000 --worker-tmp-dir /dev/shm --workers=2 --timeout 90  --access-logfile - --error-logfile - "app:create_app()"
exec gunicorn -b :5000 --worker-tmp-dir /dev/shm --workers=2 --timeout 1200 --access-logfile - --error-logfile - "app:create_app()"
