## Installation

You can either install the Python dependencies directly or run the project using Docker.

### 1. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

The required dependencies are listed in [`requirements.txt`](https://github.com/wolfofnox/pr-reviewer/blob/main/requirements.txt).

### 2. Configure API Keys

Edit [`app.py`](https://github.com/wolfofnox/pr-reviewer/blob/main/app.py) and provide your own credentials:

* **OpenAI API key**
* **GitHub Personal Access Token**

These are used to access the LLM and GitHub repositories.

## Usage

### 1. Start the API

Run the application:

```bash
python app.py
```

By default, the server starts on:

```text
http://localhost:8000
```

### 2. Create a Project

Register a GitHub repository by providing its owner and name.

Example:

```http
POST /projects
```

```json
{
    "name": "boo",
    "repo": "owner/repo",
    "docs": ["https://docs.python.org/3.14/"]
}
```

The repository documentation will be downloaded and indexed for future reviews.

### 3. Refresh Documentation (Optional)

If the repository documentation changes, refresh the indexed data:

```http
POST /projects/{project_id}/refresh
```

### 4. Submit a Pull Request for Review

Request an AI review of a pull request:

```http
POST /review
```

Provide the repository, pull request number (or commit SHA, depending on your setup), and the reviewer will:

* Download the changed files
* Retrieve relevant project documentation using RAG
* Analyze the changes with an LLM
* Return a Markdown review containing issues, suggestions, and improvement recommendations, allongside some metadata and individual responses.

The generated Markdown can be displayed directly or integrated into tools such as **n8n** to automatically comment on GitHub pull requests.
