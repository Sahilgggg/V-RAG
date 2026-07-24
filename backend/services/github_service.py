"""GitHub repository download and file parsing service."""

import io
import os
import re
import zipfile
from dataclasses import dataclass

import requests


@dataclass
class CodeFile:
    """Represents a single file from a repository."""

    path: str
    content: str
    language: str


# ─── File filtering config ────────────────────────────────────────────────────

# Extensions we index for RAG
_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".scala", ".r",
    ".html", ".css", ".scss", ".less", ".vue", ".svelte",
    ".md", ".rst", ".txt",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".sql", ".graphql", ".proto",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".xml", ".csv",
}

# Exact filenames we also index (no extension match needed)
_CODE_FILENAMES = {
    "Dockerfile", "Makefile", "Rakefile", "Gemfile", "Procfile",
    "docker-compose.yml", "docker-compose.yaml",
    ".gitignore", ".dockerignore", ".env.example",
    "requirements.txt", "setup.py", "setup.cfg", "pyproject.toml",
    "package.json", "tsconfig.json", "webpack.config.js",
    "Cargo.toml", "go.mod", "go.sum", "build.gradle", "pom.xml",
}

# Directories to skip entirely
_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".next", ".nuxt", "out", "target",
    ".idea", ".vscode", ".vs",
    "vendor", "bower_components",
    "eggs", ".eggs", "*.egg-info",
    "coverage", ".nyc_output", "htmlcov",
}

# Max sizes
MAX_FILE_SIZE_BYTES = 100 * 1024   # 100 KB per file
MAX_REPO_SIZE_MB = 50              # 50 MB total zip
MAX_FILE_COUNT = 500               # max files to index

# Extension → language mapping
_LANG_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "javascript", ".tsx": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp", ".cs": "csharp",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".kt": "kotlin",
    ".scala": "scala", ".r": "r",
    ".html": "html", ".css": "css", ".scss": "scss", ".vue": "vue",
    ".md": "markdown", ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".sql": "sql", ".sh": "shell",
    ".xml": "xml", ".graphql": "graphql", ".proto": "protobuf",
}


# ─── Repo URL parsing ─────────────────────────────────────────────────────────


def parse_github_url(url: str) -> tuple[str, str]:
    """
    Extract (owner, repo) from a GitHub URL.

    Supports:
      - https://github.com/owner/repo
      - https://github.com/owner/repo.git
      - github.com/owner/repo
      - owner/repo
    """
    url = url.strip().rstrip("/")

    # Remove .git suffix
    if url.endswith(".git"):
        url = url[:-4]

    # Try full URL pattern
    match = re.match(
        r"(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/]+)", url
    )
    if match:
        return match.group(1), match.group(2)

    # Try owner/repo shorthand
    match = re.match(r"^([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)$", url)
    if match:
        return match.group(1), match.group(2)

    raise ValueError(
        f"Could not parse GitHub URL: {url}. "
        "Expected format: https://github.com/owner/repo or owner/repo"
    )


# ─── Download ─────────────────────────────────────────────────────────────────


def download_repo_zip(
    owner: str, repo: str, github_token: str | None = None
) -> bytes:
    """Download a GitHub repo as a zip archive via the API."""
    url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
    headers = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    resp = requests.get(url, headers=headers, timeout=60, stream=True)

    if resp.status_code == 404:
        raise ValueError(
            f"Repository '{owner}/{repo}' not found. "
            "Check the URL or provide a GitHub token for private repos."
        )
    if resp.status_code == 401:
        raise ValueError("GitHub authentication failed. Check your token.")
    resp.raise_for_status()

    # Check size from Content-Length if available
    content_length = resp.headers.get("Content-Length")
    if content_length and int(content_length) > MAX_REPO_SIZE_MB * 1024 * 1024:
        raise ValueError(
            f"Repository is too large (>{MAX_REPO_SIZE_MB} MB). "
            "Try a smaller repo."
        )

    # Read the full response
    zip_bytes = resp.content
    size_mb = len(zip_bytes) / (1024 * 1024)
    if size_mb > MAX_REPO_SIZE_MB:
        raise ValueError(
            f"Repository zip is {size_mb:.1f} MB (limit: {MAX_REPO_SIZE_MB} MB)."
        )

    return zip_bytes


# ─── Parse ─────────────────────────────────────────────────────────────────────


def _should_skip_path(path: str) -> bool:
    """Check if a file path should be skipped."""
    parts = path.split("/")
    for part in parts:
        if part in _SKIP_DIRS:
            return True
    return False


def _is_indexable_file(filename: str) -> bool:
    """Check if a file should be indexed based on extension or name."""
    if filename in _CODE_FILENAMES:
        return True
    _, ext = os.path.splitext(filename)
    return ext.lower() in _CODE_EXTENSIONS


def _detect_language(filename: str) -> str:
    """Detect programming language from file extension."""
    _, ext = os.path.splitext(filename)
    return _LANG_MAP.get(ext.lower(), "text")


def parse_repo_files(zip_bytes: bytes) -> list[CodeFile]:
    """
    Extract and filter code files from a repo zip archive.

    Returns a list of CodeFile objects, sorted by path.
    """
    files: list[CodeFile] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            # Skip directories
            if info.is_dir():
                continue

            # GitHub zips have a top-level dir like "owner-repo-hash/"
            # Strip it to get the real repo path
            parts = info.filename.split("/", 1)
            if len(parts) < 2:
                continue
            rel_path = parts[1]

            if not rel_path:
                continue

            # Skip unwanted directories
            if _should_skip_path(rel_path):
                continue

            # Skip non-code files
            filename = os.path.basename(rel_path)
            if not _is_indexable_file(filename):
                continue

            # Skip large files
            if info.file_size > MAX_FILE_SIZE_BYTES:
                continue

            # Read and decode
            try:
                raw = zf.read(info.filename)
                content = raw.decode("utf-8", errors="replace")
            except Exception:
                continue

            # Skip empty files
            if not content.strip():
                continue

            files.append(
                CodeFile(
                    path=rel_path,
                    content=content,
                    language=_detect_language(filename),
                )
            )

            # Cap file count
            if len(files) >= MAX_FILE_COUNT:
                break

    files.sort(key=lambda f: f.path)
    return files


# ─── High-level API ───────────────────────────────────────────────────────────


def load_repo(
    repo_url: str, github_token: str | None = None
) -> list[CodeFile]:
    """
    Download and parse a GitHub repository.

    Args:
        repo_url: GitHub URL or owner/repo shorthand.
        github_token: Optional GitHub PAT for private repos.

    Returns:
        List of CodeFile objects ready for indexing.
    """
    owner, repo = parse_github_url(repo_url)
    zip_bytes = download_repo_zip(owner, repo, github_token)
    files = parse_repo_files(zip_bytes)

    if not files:
        raise ValueError(
            f"No indexable code files found in '{owner}/{repo}'. "
            "The repo may be empty or contain only binary files."
        )

    return files
