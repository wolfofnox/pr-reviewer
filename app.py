from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from github import Github
import uuid
import os
import json
import io
import zipfile
import tempfile
import shutil
import requests
import logging
from urllib.parse import urlparse, urljoin
from collections import deque
import re
import openai
from openai import OpenAI
import chromadb

PROJECT_ROOT = "/projects"

# GitHub token (used for PyGithub and direct API downloads)
GITHUB_TOKEN = "YOUR_KEY"
g = Github(GITHUB_TOKEN)

OPENAI_API_KEY = "YOUR_KEY"

app = FastAPI(
    title="PR Reviewer API",
    version="0.1.0"
)

# logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger("pr_reviewer")
logger.setLevel(logging.DEBUG)


# Create or open Chroma client with local persistence
chroma_client = chromadb.PersistentClient(path='/vectordb')

# -------------------------
# Models
# -------------------------

class ProjectCreate(BaseModel):
    name: str
    repo: str
    docs: list[str] = []
    notion_page: str | None = None


class ReviewRequest(BaseModel):
    pr: int
    chat_model: str | None = None
    embeddings_model:str | None = None


class QueryRequest(BaseModel):
    prompt: str
    top_k: int = 5
    model: str | None = None
    
# -------------------------
# Utils
# -------------------------

def proj_dir(name: str) -> str:
    return f"{PROJECT_ROOT}/{name}"

def base_dir(name: str) -> str:
    return f"{PROJECT_ROOT}/{name}/base"

def head_dir(name: str) -> str:
    return f"{PROJECT_ROOT}/{name}/head"

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _normalize_repo_name(repo: str) -> str:
    """Normalize various repo inputs to `owner/repo`.

    Handles inputs like:
    - owner/repo
    - https://github.com/owner/repo
    - https://github.com/owner/repo/ (with extra path segments)
    - git@github.com:owner/repo.git
    - owner/repo.git
    """
    if not repo:
        return repo
    repo = repo.strip()

    # SSH form: git@github.com:owner/repo.git
    if repo.startswith("git@"):
        path = repo.split(":", 1)[-1]
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2:
            owner = parts[0]
            name = parts[1].removesuffix('.git') if hasattr(parts[1], 'removesuffix') else parts[1].rstrip('.git')
            return f"{owner}/{name}"
        return path.removesuffix('.git') if hasattr(path, 'removesuffix') else path.rstrip('.git')

    # HTTP(S) form
    if repo.startswith("http"):
        parsed = urlparse(repo)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2:
            owner = parts[0]
            name = parts[1].removesuffix('.git') if hasattr(parts[1], 'removesuffix') else parts[1].rstrip('.git')
            return f"{owner}/{name}"

    # Plain form: owner/repo or owner/repo/extra
    parts = [p for p in repo.split("/") if p]
    if len(parts) >= 2:
        owner = parts[0]
        name = parts[1].removesuffix('.git') if hasattr(parts[1], 'removesuffix') else parts[1].rstrip('.git')
        return f"{owner}/{name}"

    # Fallback: strip .git if present
    return repo.removesuffix('.git') if hasattr(repo, 'removesuffix') else repo.rstrip('.git')


def save_project_metadata(project_id: str, metadata: dict):
    project_path = proj_dir(project_id)
    ensure_dir(project_path)
    path = os.path.join(project_path, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)


def load_project_metadata(project_id: str) -> dict | None:
    path = os.path.join(proj_dir(project_id), "metadata.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def download_repo_snapshot(repo_full_name: str, ref: str, target_dir: str):
    """Download a zipball of the repo at ref and extract into target_dir (keeps metadata.json)."""
    ensure_dir(target_dir)

    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    url = g.get_repo(repo_full_name).get_archive_link(archive_format="zipball", ref=ref)
    resp = requests.get(url, headers=headers, stream=True, timeout=30)
    resp.raise_for_status()

    # Extract to a temporary directory first
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = io.BytesIO(resp.content)
        with zipfile.ZipFile(buf) as z:
            # Zip usually contains a single top-level folder; strip it
            for member in z.infolist():
                name = member.filename
                parts = name.split('/', 1)
                if len(parts) == 1:
                    relpath = parts[0]
                else:
                    relpath = parts[1]
                if not relpath:
                    continue
                dest_path = os.path.join(tmpdir, relpath)
                if member.is_dir():
                    ensure_dir(dest_path)
                    continue
                ensure_dir(os.path.dirname(dest_path))
                with z.open(member) as src, open(dest_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        # Remove existing files in target_dir except metadata.json
        for root, dirs, files in os.walk(target_dir):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                except Exception:
                    pass
            for name in dirs:
                dirpath = os.path.join(root, name)
                if os.path.exists(dirpath) and not any(p.startswith(os.path.join(dirpath, 'metadata.json')) for p in [os.path.join(root, f) for f in files]):
                    try:
                        shutil.rmtree(dirpath)
                    except Exception:
                        pass

        # Move extracted files into target_dir
        for item in os.listdir(tmpdir):
            s = os.path.join(tmpdir, item)
            d = os.path.join(target_dir, item)
            if os.path.exists(d):
                if os.path.isdir(d):
                    shutil.rmtree(d)
                else:
                    os.remove(d)
            shutil.move(s, d)


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200):
    """Simple character-based sliding window chunker."""
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + max_chars, L)
        chunk = text[start:end]
        chunks.append((chunk, start, end))
        if end == L:
            break
        start = max(end - overlap, start + 1)
    return chunks


def index_dir(path: str, collection_id: str, model: str = "text-embedding-3-small") -> dict:
    api_key = OPENAI_API_KEY
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in environment")
    openai_client = OpenAI(api_key=api_key)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"nothing found at {path}")

    # Gather chunks
    items = []  # tuples of (text, path, start, end)
    max_file_size = 1_000_000  # skip files larger than 1MB
    for root, dirs, files in os.walk(path):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                if os.path.getsize(fpath) > max_file_size:
                    logger.debug("Skipping large file %s", fpath)
                    continue
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except Exception as e:
                logger.debug("Skipping unreadable file %s: %s", fpath, e)
                continue

            relative = os.path.relpath(fpath, path)
            chunks = chunk_text(text)
            for chunk, start, end in chunks:
                items.append({
                    "text": chunk,
                    "path": relative,
                    "start": start,
                    "end": end,
                })

    try:
        # delete existing collection if present to recreate
        _ = chroma_client.get_collection(name=collection_id)
        chroma_client.delete_collection(name=collection_id)
    except Exception:
        pass
    collection = chroma_client.create_collection(name=collection_id)

    # Batch embed and upsert into Chroma
    batch_size = 256
    total = 0
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        texts = [it["text"] for it in batch]
        try:
            resp = openai_client.embeddings.create(input=texts, model=model)
        except Exception as e:
            logger.exception("OpenAI embedding failed: %s", e)
            raise HTTPException(status_code=500, detail=f"embedding failed: {e}")

        embeddings = [r.embedding for r in resp.data]
        ids = [str(uuid.uuid4()) for _ in batch]
        metadatas = [{"path": it["path"], "start": it["start"], "end": it["end"]} for it in batch]

        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        total += len(batch)

    logger.info("Indexed %d chunks for directory %s into Chroma", total, path)

    return {"last_refresh": datetime.now().isoformat(), "indexed_chunks": total}


def _extract_text_from_html(html: str) -> str:
    # remove script/style
    html = re.sub(r"(?is)<script.*?>.*?</script>", "", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", "", html)
    # remove tags
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    # unescape HTML entities
    try:
        from html import unescape
        text = unescape(text)
    except Exception:
        pass
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_and_index_docs(project_id: str, base_url: str, max_pages: int = 50, embeddings_model: str = "text-embedding-3-small") -> dict:
    """Crawl pages under base_url (same netloc) up to max_pages and index into Chroma collection named project_id."""
    if not base_url:
        raise HTTPException(status_code=400, detail="base_url required")

    parsed = urlparse(base_url)
    base_netloc = parsed.netloc

    visited = set()
    to_visit = deque([base_url])
    pages = []

    session = requests.Session()
    session.headers.update({"User-Agent": "pr-reviewer-bot/1.0"})

    while to_visit and len(visited) < max_pages:
        url = to_visit.popleft()
        if url in visited:
            continue
        try:
            resp = session.get(url, timeout=10)
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            logger.debug("Failed to fetch %s: %s", url, e)
            visited.add(url)
            continue

        visited.add(url)
        text = _extract_text_from_html(html)
        if text:
            pages.append({"url": url, "text": text})

        # find links
        for href in re.findall(r'href=["\'](.*?)["\']', html):
            next_url = urljoin(url, href)
            np = urlparse(next_url)
            if np.scheme.startswith("http") and np.netloc == base_netloc:
                # normalize
                norm = np.scheme + "://" + np.netloc + np.path
                if norm not in visited and norm not in to_visit:
                    to_visit.append(norm)

    # prepare collection
    try:
        coll = chroma_client.get_collection(name=project_id)
    except Exception:
        coll = chroma_client.create_collection(name=project_id)

    # convert pages to chunks and index
    items = []
    for p in pages:
        chunks = chunk_text(p["text"])
        if not chunks:
            items.append({"text": p["text"], "meta": {"source": p["url"]}})
        else:
            for idx, (chunk, start, end) in enumerate(chunks):
                items.append({"text": chunk, "meta": {"source": p["url"], "chunk": idx, "start": start, "end": end}})

    # embed & add
    api_key = OPENAI_API_KEY
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    openai_client = OpenAI(api_key=api_key)

    batch_size = 256
    total = 0
    model = embeddings_model or "text-embedding-3-small"
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        texts = [it["text"] for it in batch]
        try:
            resp = openai_client.embeddings.create(input=texts, model=model)
        except Exception as e:
            logger.exception("Embedding failed while indexing docs: %s", e)
            raise HTTPException(status_code=500, detail=f"embedding failed: {e}")
        embeddings = [r.embedding for r in resp.data]
        ids = [str(uuid.uuid4()) for _ in batch]
        metadatas = [it["meta"] for it in batch]
        coll.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        total += len(batch)

    logger.info("Indexed %d doc-chunks for project %s from %s", total, project_id, base_url)
    return {"indexed_chunks": total, "pages_fetched": len(pages)}

def query_db(collection_id: str, request: str, model: str, top_k: int):
    logger.info("Query requested for collecton %s", collection_id)

    if not request:
        return {}

    model = model or "text-embedding-3-small"
    api_key = OPENAI_API_KEY
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    openai_client = OpenAI(api_key=api_key)

    try:
        emb_resp = openai_client.embeddings.create(input=[request], model=model)
        q_emb = emb_resp.data[0].embedding
    except Exception as e:
        logger.exception("Embedding failed for query: %s", e)
        raise HTTPException(status_code=500, detail=f"embedding failed: {e}")
    
    try:
        coll = chroma_client.get_collection(name=collection_id)
    except Exception as e:
        logger.exception(f"Chroma collection '{collection_id}' access failed: %s", e)
        raise HTTPException(status_code=404, detail=f"vector collection '{collection_id}' not found")

    try:
        res = coll.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances"])
    except Exception as e:
        logger.exception("Chroma query failed: %s", e)
        raise HTTPException(status_code=500, detail=f"vector query failed: {e}")

    docs = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    hits = []
    for doc, meta, dist in zip(docs, metadatas, distances):
        hits.append({
            "text": doc,
            "metadata": meta,
            "distance": dist,
        })

    return {"results": hits}

# -------------------------
# Projects
# -------------------------

@app.get("/api/projects")
def list_projects():
    projects = []
    if os.path.exists(PROJECT_ROOT):
        for name in os.listdir(PROJECT_ROOT):
            metadata = load_project_metadata(name)
            if metadata:
                projects.append({
                    "project_id": name,
                    "name": metadata.get("name"),
                    "repo": metadata.get("repo")
                })
    return {"projects": projects}

@app.post("/api/projects")
def create_project(project: ProjectCreate):

    project_id = project.name.lower().replace(" ", "-")
    # If project already exists, return an error
    existing = load_project_metadata(project_id)
    if existing is not None or os.path.exists(proj_dir(project_id)):
        logger.warning("Create failed: project %s already exists", project_id)
        raise HTTPException(status_code=400, detail=f"project '{project_id}' already exists")
    # persist a small metadata file for this project so we can refresh later
    metadata = {
        "name": project.name,
        "repo": _normalize_repo_name(project.repo),
        "docs": project.docs,
        "notion_page": project.notion_page,
        "last_refrest": datetime.now().isoformat()
    }
    save_project_metadata(project_id, metadata)

    logger.info("Created project %s -> %s", project.name, project_id)

    for doc in metadata.get("docs", []) or []:
        try:
            fetch_and_index_docs(project_id, doc, max_pages=50, embeddings_model="text-embedding-3-small")
        except Exception:
            logger.exception("Failed to fetch and index docs for %s", doc)

    return {
        "status": "completed",
        "project_id": project_id,
        "meta": metadata
    }

@app.post("/api/projects/{project_id}/refresh")
def refresh_project(project_id: str):
    logger.info("Refresh requested for project %s", project_id)
    # Load metadata to know which repo to refresh
    metadata = load_project_metadata(project_id)
    if not metadata:
        logger.warning("Refresh failed: project %s not found", project_id)
        raise HTTPException(status_code=404, detail=f"project '{project_id}' not found")

    try:
        for doc in metadata.get("docs", []) or []:
            try:
                fetch_and_index_docs(project_id, doc, max_pages=50, embeddings_model="text-embedding-3-small")
            except Exception:
                logger.exception("Failed to fetch and index docs for %s", doc)
    except Exception as e:
        logger.exception("Refresh failed for project %s: %s", project_id, str(e))
        return {"project_id": project_id, "status": "error", "message": str(e)}
    
    metadata.update({"last_refrest": datetime.now().isoformat()})
    save_project_metadata(project_id, metadata)

    return {
        "project_id": project_id,
        "status": "completed"
    }

@app.get("/api/projects/{project_id}/status")
def project_status(project_id: str):
    metadata = load_project_metadata(project_id)
    if not metadata:
        logger.warning("Refresh failed: project %s not found", project_id)
        raise HTTPException(status_code=404, detail=f"project '{project_id}' not found")

    return {
        "project_id": project_id,
        "state": "ready",
        "last_refresh": metadata.get("last_refresh")
    }



# -------------------------
# PR Review
# -------------------------

@app.post("/api/projects/{project_id}/review")
def review_project(project_id: str, request: ReviewRequest):
    logger.info("Starting review for project %s pr=%s", project_id, request.pr)

    # Load project metadata
    metadata = load_project_metadata(project_id)
    if not metadata:
        logger.warning("Review failed: project %s not found", project_id)
        raise HTTPException(status_code=404, detail=f"project '{project_id}' not found")

    repo_full = metadata.get("repo")
    if not repo_full:
        raise HTTPException(status_code=500, detail="project metadata missing repo")

    # Resolve PR and SHAs
    try:
        repo = g.get_repo(repo_full)
        pr_obj = repo.get_pull(request.pr)
        base_sha = pr_obj.base.sha
        head_sha = pr_obj.head.sha
        # map PR files for patch access
        pr_files_map = {f.filename: f for f in pr_obj.get_files()}
    except Exception as e:
        logger.exception("Failed to load PR info: %s", e)
        raise HTTPException(status_code=500, detail=f"failed to load PR {request.pr}: {e}")

    # Download snapshots
    bdir = base_dir(project_id)
    hdir = head_dir(project_id)
    ensure_dir(bdir)
    ensure_dir(hdir)
    try:
        download_repo_snapshot(repo_full, base_sha, bdir)
        download_repo_snapshot(repo_full, head_sha, hdir)
    except Exception as e:
        logger.exception("Snapshot download failed: %s", e)
        raise HTTPException(status_code=500, detail=f"snapshot download failed: {e}")

    # Index base and head into temporary collections
    base_col = f"{project_id}-base"
    head_col = f"{project_id}-head"
    try:
        index_dir(bdir, base_col)
        index_dir(hdir, head_col)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Indexing snapshots failed: %s", e)
        raise HTTPException(status_code=500, detail=f"indexing failed: {e}")

    # Compute changed files by comparing file contents
    changed = []  # list of dicts {path, base_text, head_text}
    files_seen = set()
    for root, dirs, files in os.walk(bdir):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), bdir)
            files_seen.add(rel)
    for root, dirs, files in os.walk(hdir):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), hdir)
            files_seen.add(rel)

    for rel in sorted(files_seen):
        bpath = os.path.join(bdir, rel)
        hpath = os.path.join(hdir, rel)
        btext = None
        htext = None
        try:
            if os.path.exists(bpath):
                with open(bpath, "r", encoding="utf-8", errors="replace") as f:
                    btext = f.read()
        except Exception:
            btext = None
        try:
            if os.path.exists(hpath):
                with open(hpath, "r", encoding="utf-8", errors="replace") as f:
                    htext = f.read()
        except Exception:
            htext = None

        if btext == htext:
            continue
        changed.append({"path": rel, "base": btext or "", "head": htext or ""})

    logger.info("Found %d changed files in PR %s", len(changed), request.pr)

    # Simple symbol splitter: splits on lines that look like python/JS function/class definitions
    import re

    def split_symbols(text: str):
        """Split text into symbol chunks and record start/end line numbers.

        Returns a list of dicts: {name, code, start_line, end_line}.
        Also emits debug logs showing the symbol name and its line range.
        """
        if not text:
            return [{"name": "<empty>", "code": ""}]
        lines = text.splitlines(keepends=True)
        pattern = re.compile(r"^\s*(def|class|function)\s+([A-Za-z0-9_]+)")
        chunks = []
        cur_name = None
        cur_lines = []
        cur_start = None
        line_no = 1
        for ln in lines:
            m = pattern.match(ln)
            if m:
                if cur_name is not None:
                    end_line = line_no - 1
                    code = "".join(cur_lines)
                    chunks.append({"name": cur_name, "code": code, "start_line": cur_start, "end_line": end_line})
                    logger.debug("split_symbols: name=%s start=%d end=%d", cur_name, cur_start, end_line)
                cur_name = m.group(2)
                cur_lines = [ln]
                cur_start = line_no
            else:
                if cur_name is None:
                    cur_name = "<top>"
                    cur_lines = [ln]
                    cur_start = line_no
                else:
                    cur_lines.append(ln)
            line_no += 1
        if cur_name is not None:
            end_line = line_no - 1
            code = "".join(cur_lines)
            chunks.append({"name": cur_name, "code": code, "start_line": cur_start, "end_line": end_line})
            logger.debug("split_symbols: name=%s start=%d end=%d", cur_name, cur_start, end_line)
        if not chunks:
            logger.debug("no symbols found, returning entire text")
            return [{"name": "<file>", "code": text}]
        return chunks

    def extract_added_blocks_from_patch(patch: str) -> list:
        """Return a list of added-text blocks from a unified diff patch (lines starting with '+', excluding headers)."""
        if not patch:
            return []
        added_blocks = []
        cur = []
        for ln in patch.splitlines():
            if ln.startswith('+++') or ln.startswith('---') or ln.startswith('@@'):
                # diff metadata; treat as boundary
                if cur:
                    added_blocks.append('\n'.join(cur))
                    cur = []
                continue
            if ln.startswith('+') and not ln.startswith('+++'):
                cur.append(ln[1:])
            else:
                if cur:
                    added_blocks.append('\n'.join(cur))
                    cur = []
        if cur:
            added_blocks.append('\n'.join(cur))
        return added_blocks

    def get_changed_symbols_for_file(file_path: str, base_text: str, head_text: str, pr_files_map: dict):
        """Return list of symbol dicts (name, code) that changed in this file using the PR patch when available."""
        # try to find the PR file entry
        pf = pr_files_map.get(file_path)
        if pf is None:
            # no patch available; fallback to any symbols in head_text
            return split_symbols(head_text)

        # if file was removed, nothing to review
        if getattr(pf, 'status', None) == 'removed':
            return []

        patch = getattr(pf, 'patch', None)
        if not patch:
            # for added files, review all symbols in head
            if getattr(pf, 'status', None) == 'added':
                return split_symbols(head_text)
            return split_symbols(head_text)

        added_blocks = extract_added_blocks_from_patch(patch)
        symbols = []
        logger.info("Found %d added blocks in file %s", len(added_blocks), path)
        for block in added_blocks:
            logger.debug("New block:")
            # find symbols inside added block
            syms = split_symbols(block)
            for s in syms:
                # avoid duplicates by name+code
                key = (s.get('name'), s.get('code'))
                symbols.append(s)
        # If no symbols found in added hunks and file was added, include all head symbols
        if not symbols and getattr(pf, 'status', None) == 'added':
            return split_symbols(head_text)
        # dedupe by name+code while preserving order
        seen = set()
        unique = []
        for s in symbols:
            k = (s.get('name'), s.get('code'))
            if k in seen:
                continue
            seen.add(k)
            unique.append(s)
        return unique

    findings = []

    # prepare OpenAI client for review prompts
    chat_model = request.chat_model or "gpt-4o-mini"
    embeddings_model = request.embeddings_model or "text-embedding-3-small"
    api_key = OPENAI_API_KEY
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    openai_client = OpenAI(api_key=api_key)

    # collections to search: project docs (if exists), base, head
    collections_to_search = []
    try:
        chroma_client.get_collection(name=project_id)
        collections_to_search.append(project_id)
    except Exception:
        pass
    collections_to_search.extend([base_col, head_col])

    for ch in changed:
        path = ch["path"]
        head_text = ch["head"]
        symbols = get_changed_symbols_for_file(path, ch.get("base", ""), head_text, pr_files_map)
        if not symbols:
            logger.info("No changed symbols found in file %s, skipping", path)
            continue
        logger.info("Found %d changed symbols in file %s", len(symbols), path)

        for sym in symbols:
            sym_name = sym.get("name")
            code_snippet = sym.get("code")

            # gather search hits across collections
            aggregated_hits = []
            for col in collections_to_search:
                try:
                    resp = query_db(col, str(code_snippet)[:2000], model=embeddings_model, top_k=5)
                    hits = resp.get("results", [])
                except Exception:
                    hits = []
                aggregated_hits.append({"collection": col, "hits": hits})

            # build prompt for reviewer
            prompt = (
                f"You are a helpful code reviewer.\nFile: {path}\nSymbol: {sym_name}\n\n"
                f"Provide concise findings about potential bugs, style issues, security concerns, and suggestions.\n\n"
                f"Code snippet:\n```{str(code_snippet)[:8000]}```\n\n"
                f"Context search hits (truncated): {json.dumps(aggregated_hits, default=str)[:4000]}"
            )

            try:
                # call the chat completion API
                chat_resp = openai_client.chat.completions.create(
                    model=chat_model,
                    messages=[
                        {"role": "system", "content": "You are a senior software engineer and code reviewer."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800
                )
                review_text = None
                if hasattr(chat_resp, "choices") and len(chat_resp.choices) > 0:
                    review_text = chat_resp.choices[0].message.content if chat_resp.choices[0].message else None
                # fallback for other shapes
                if not review_text:
                    review_text = getattr(chat_resp, "text", None) or str(chat_resp)
            except Exception as e:
                logger.exception("OpenAI review call failed: %s", e)
                review_text = f"review failed: {e}"

            findings.append({
                "file": path,
                "symbol": sym_name,
                "review": review_text,
                "search_hits": aggregated_hits
            })

    # Final pass: synthesize a human-readable overall review from findings
    def _summarize_findings(findings_list, model_name, openai_client):
        if not findings_list:
            return "No findings detected."
        # build a compact representation for the prompt
        items = []
        for f in findings_list[:40]:
            review = f.get("review") or ""
            review_snip = (review[:800] + "...") if len(review) > 800 else review
            items.append(f"File: {f.get('file')}\nSymbol: {f.get('symbol')}\nReview: {review_snip}")
        payload = "\n\n".join(items)

        prompt = (
            "You are an expert senior software engineer. Convert the following raw review findings into a clear, human-friendly code review summary. "
            "Group issues by file, highlight severity (Critical/High/Medium/Low), and provide actionable suggestions. Keep it concise and readable.\n\n"
            f"Raw findings:\n{payload}\n\nSummary:\n"
        )

        try:
            resp = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a senior software engineer and code reviewer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800
            )
            summary = None
            if hasattr(resp, "choices") and len(resp.choices) > 0:
                summary = resp.choices[0].message.content if resp.choices[0].message else None
            if not summary:
                summary = getattr(resp, "text", None) or str(resp)
            return summary
        except Exception as e:
            logger.exception("Failed to synthesize summary: %s", e)
            return f"(summary generation failed: {e})"

    summary_text = _summarize_findings(findings, chat_model, openai_client)

    # cleanup temporary collections
    try:
        chroma_client.delete_collection(name=base_col)
    except Exception:
        pass
    try:
        chroma_client.delete_collection(name=head_col)
    except Exception:
        pass

    return {"summary": summary_text, "findings": findings}
