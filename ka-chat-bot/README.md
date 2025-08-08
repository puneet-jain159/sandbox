---
title: "Knowledge Assistant Chatbot"
language: python
author: "Taiga Matsumoto"
date: 2025-05-26
---

# Databricks Knowledge Assistant Chatbot Application

Chat applications powered by your Databricks Knowledge Assistant

## Features

- 🚀 Real-time chat interface
- 💾 Chat history persistence
- 🔄 Message regeneration capability
- ⚡ Streaming responses
- 🔒 On-behalf-of-user authentication
- 🎯 Rate limiting and error handling

## Architecture

The application is built with:
- FastAPI for the backend API
- PostgreSQL (via Databricks Database) for chat history storage
- SQLite support for local development
- React frontend


## Getting Started

### Quick Setup (Recommended)

Use the Makefile for easy setup and management:

```bash
# Complete setup (installs uv, creates venv, installs all dependencies)
make setup

# Run development server
make dev

# Run production server
make run
```

### Manual Setup

1. Clone the repository
2. Set up environment variables:
   ```bash
   # Copy the example environment file
   cp env.example .env
   
   # Edit .env with your configuration:
   # - LOCAL_API_TOKEN: your PAT used only for local development
   # - DATABRICKS_HOST: your Databricks domain url (e.g. "your-domain.cloud.databricks.com")
   # - SERVING_ENDPOINT_NAME: your Knowledge Assistant's serving endpoint (e.g "ka-123-endpoint")
   ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Build the frontend

    [1]. Navigate to the frontend directory:

    ```bash
    cd frontend
    ```

    [2]. Install dependencies:

    ```bash
    npm install
    ```
    [3a]. Generate a local build:

    ```bash
    npm run build
    ```

    [3b]. Generate a production build for app deployment:

    ```bash
    npm run build:prod
    ```

5. Run the server:
    ```bash
    python main.py
    ```

## Database Configuration

The application supports both SQLite (for local development) and PostgreSQL (via Databricks Database) for production.

### PostgreSQL Setup (Recommended)

1. **Configure database settings**:
   ```bash
   make db-setup
   ```

2. **Add to your `.env` file**:
   ```
   DB_USERNAME=your-username@databricks.com
   DB_INSTANCE_NAME=your-instance-name
   CLIENT_ID=your-client-id
   CLIENT_SECRET=your-client-secret
   ```

3. **Validate configuration**:
   ```bash
   make db-validate
   ```

4. **Migrate from SQLite (if needed)**:
   ```bash
   make db-migrate
   ```

### SQLite (Local Development)

For local development, the application will automatically use SQLite if PostgreSQL configuration is not provided.

## Makefile Commands

The project includes a comprehensive Makefile for easy development:

- `make setup` - Complete project setup (install uv, create venv, install deps)
- `make install` - Install all dependencies
- `make install-uv` - Install uv package manager
- `make install-backend` - Install backend dependencies using uv
- `make install-frontend` - Install frontend dependencies
- `make dev` - Run development server with hot reload
- `make run` - Run production server with gunicorn
- `make test` - Run tests
- `make clean` - Clean up generated files
- `make lint` - Run linting with ruff
- `make format` - Format code with ruff
- `make check-deps` - Check for outdated dependencies
- `make info` - Show project information

## Key Components

- `fronted/`: React frontend
- `main.py`: FastAPI application entry point
- `utils/`: Helper functions and utilities
- `models.py`: Data models and schemas
- `chat_database.py`: Database interactions
