#!/bin/bash
source venv/bin/activate
exec rq worker toxpro-tasks --name toxpro-tasks --url redis://toxpro-redis-1:6379