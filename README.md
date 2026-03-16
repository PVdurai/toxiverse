# ToxiVerse Project

This project runs a **Flask-based application** with **Docker and Redis Queue (RQ)** support for background task processing.

Follow the steps below to set up the environment and run the project.

---

# Prerequisites

Make sure the following software is installed on your system:

* Conda (Anaconda or Miniconda)
* Docker Desktop or Docker Engine
* Python 3.11

---

# 1. Create Conda Environment

Create a new conda environment named `toxpro`.

```bash
conda create -n toxpro python=3.11
```

Activate the environment:

```bash
conda activate toxpro
```

---

# 2. Install Python Dependencies

Navigate to the project root directory and install the required dependencies.

```bash
pip install -r requirements.txt
```

---

# 3. Start Docker Services

Ensure Docker is installed and running.

Run the docker compose file:

```bash
docker compose -f docker-compose.yml -p toxpro up -d
```

---

# 4. Start Redis Server

Run Redis using Docker:

```bash
docker run --name my-redis -p 6379:6379 redis
```

---

# 5. Start the RQ Worker

In a separate terminal, activate the environment and start the worker:

```bash
conda activate toxpro
rq worker toxpro-tasks
```

---

# 6. Run the Flask Application

Start the Flask development server:

```bash
flask run
```