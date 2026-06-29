## Installation

You can either install the Python dependencies directly or run the project using Docker.

### Option 1: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

The required dependencies are listed in [`requirements.txt`](https://github.com/wolfofnox/pr-reviewer/blob/main/requirements.txt).

---

### Option 2: Run with Docker Compose (Recommended)

The project includes a Docker Compose setup. The `ollama` and `n8n` services are optional—the PR Reviewer can run independently if you already have an LLM endpoint and do not use the n8n workflow.

```yaml
services:

  ollama:
    image: ollama/ollama
    container_name: ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - /data/ollama:/root/.ollama
    gpus: all
    networks:
      - ai-network

  n8n:
    image: n8nio/n8n
    container_name: n8n
    restart: unless-stopped
    ports:
      - "5678:5678"
    environment:
      - N8N_HOST=n8n
      - N8N_PORT=5678
      - N8N_PROTOCOL=http
      - GENERIC_TIMEZONE=Europe/Prague
      - N8N_SECURE_COOKIE=false
    volumes:
      - ./n8n:/home/node/.n8n
    depends_on:
      - ollama
    networks:
      - ai-network

  pr-reviewer:
    build:
      context: ./pr-reviewer
    container_name: pr-reviewer
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./pr-reviewer:/app
      - /data/pr-reviewer/projects:/projects
    networks:
      - ai-network

networks:
  ai-network:
    driver: bridge
```

Start all configured services:

```bash
docker compose up -d
```

Or, if you only want to run the API:

```bash
docker compose up -d pr-reviewer
```
