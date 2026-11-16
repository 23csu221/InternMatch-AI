import re
import requests

TIMEOUT = 25

class JoobleAPIError(Exception):
    pass

def _clean_html(text):
    text = re.sub(r"<[^>]+>", " ", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()

def search_jooble(key, keywords, location, country_code="in"):
    """
    Jooble REST API search. Regional API keys must use their matching country domain.
    India key => https://in.jooble.org/api/{key}
    """
    if not key or not str(key).strip():
        raise JoobleAPIError("Jooble API key is missing. Add it to .streamlit/secrets.toml.")
    country_code = (country_code or "in").lower().strip()
    endpoint = f"https://{country_code}.jooble.org/api/{key.strip()}"
    payload = {
        "keywords": keywords,
        "location": location,
        "page": "1",
        "ResultOnPage": "30",
        "companysearch": "false",
    }
    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT,
        )
    except requests.RequestException:
        raise JoobleAPIError("Could not reach Jooble. Check your internet connection and try again.") from None

    if response.status_code == 403:
        raise JoobleAPIError(
            "Jooble returned 403 Access Denied. Its documentation says this means the key is invalid/missing or belongs to a different country domain. "
            "Because this project uses an India key, it now calls in.jooble.org. If this continues, request/confirm an India-specific key with Jooble."
        )
    if response.status_code == 429:
        raise JoobleAPIError("Jooble rate limit reached (HTTP 429). Wait for the quota to reset or contact Jooble about your API limit.")
    if response.status_code == 404:
        raise JoobleAPIError("Jooble returned 404. The regional API endpoint was not found; check the country domain with Jooble.")
    if response.status_code >= 400:
        raise JoobleAPIError(f"Jooble request failed with HTTP {response.status_code}. The API key is deliberately hidden from this error message.")
    try:
        payload = response.json()
    except ValueError:
        raise JoobleAPIError("Jooble returned a response that was not valid JSON.") from None

    jobs = []
    for item in payload.get("jobs", []):
        link = item.get("link") or item.get("url") or ""
        if not str(link).startswith("http"):
            continue
        jobs.append({
            "id": str(item.get("id") or link),
            "title": item.get("title") or "Internship",
            "company": item.get("company") or "Company not provided",
            "location": item.get("location") or location,
            "description": _clean_html(item.get("snippet") or item.get("description")),
            "salary": item.get("salary") or "",
            "job_type": item.get("type") or "",
            "url": link,
            "source": item.get("source") or "Jooble",
            "updated": item.get("updated") or "",
        })
    return jobs
