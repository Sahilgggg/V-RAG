"""YouTube transcript fetching service.

Dual-strategy approach:
  1. Supadata managed API  — reliable on cloud servers.
  2. youtube-transcript-api — fallback, works locally but often blocked on cloud IPs.
"""

import re
import requests


def extract_video_id(url_or_id: str) -> str | None:
    """Extract a YouTube video ID from various URL formats or a raw 11-char ID."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/|/embed/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return None


def fetch_transcript_supadata(video_id: str, api_key: str) -> str:
    """Fetch transcript via the Supadata managed API."""
    url = "https://api.supadata.ai/v1/youtube/transcript"
    params = {"videoId": video_id}
    headers = {"x-api-key": api_key}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "content" in data:
        return " ".join(seg["text"] for seg in data["content"] if "text" in seg)
    raise ValueError("Unexpected Supadata response format")


def fetch_transcript_youtube_api(video_id: str) -> str:
    """Fallback: use youtube-transcript-api (may fail on cloud IPs)."""
    from youtube_transcript_api import YouTubeTranscriptApi

    ytt = YouTubeTranscriptApi()
    transcript = ytt.fetch(video_id)
    return " ".join(snippet.text for snippet in transcript)


def load_transcript(video_input: str, supadata_key: str | None = None) -> str:
    """
    Load a YouTube video transcript as a single text string.

    Tries Supadata first (if key provided), then falls back to youtube-transcript-api.
    Returns the full transcript text.
    """
    video_id = extract_video_id(video_input)
    if not video_id:
        raise ValueError(f"Could not extract a valid video ID from: {video_input}")

    errors: list[str] = []

    # Strategy 1 – Supadata (reliable on cloud)
    if supadata_key:
        try:
            text = fetch_transcript_supadata(video_id, supadata_key)
            if text.strip():
                return text
        except Exception as e:
            errors.append(f"Supadata: {e}")

    # Strategy 2 – youtube-transcript-api (local / direct)
    try:
        text = fetch_transcript_youtube_api(video_id)
        if text.strip():
            return text
    except Exception as e:
        errors.append(f"youtube-transcript-api: {e}")

    raise RuntimeError(
        "Could not fetch transcript with any method.\n" + "\n".join(errors)
    )
