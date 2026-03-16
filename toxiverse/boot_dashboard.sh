#!/bin/bash
source venv/bin/activate
exec rq-dashboard --port 5555 --redis-url redis://redis:6379