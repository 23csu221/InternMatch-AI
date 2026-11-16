import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _tokens(text):
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]*", str(text or "").lower()))

def _similarity(a, b):
    if not str(a).strip() or not str(b).strip():
        return 0.0
    try:
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1,2)).fit_transform([str(a), str(b)])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except ValueError:
        return 0.0

def rank_jobs(jobs, profile, feedback=None):
    feedback = feedback or {}
    student_skills = _tokens(profile.get("skills", ""))
    interests = profile.get("interests", "") + " " + profile.get("domains", "")
    profile_text = " ".join([
        profile.get("skills", ""), profile.get("interests", ""), profile.get("domains", ""),
        profile.get("degree", ""), profile.get("projects", ""), profile.get("certifications", "")
    ])
    ranked = []
    for original in jobs:
        job = dict(original)
        description = job.get("description", "")
        text = job.get("title", "") + " " + description
        job_tokens = _tokens(text + " " + job.get("skills", ""))
        overlap = len(student_skills & job_tokens) / max(1, len(student_skills))
        semantic = _similarity(profile_text, text)
        interest = _similarity(interests, job.get("title", "") + " " + description)
        score = 100 * (0.50 * semantic + 0.30 * overlap + 0.20 * interest)
        fb = feedback.get(job.get("id") or job.get("url"))
        if fb == "like": score = min(100, score + 4)
        elif fb == "dislike": score = max(0, score - 12)
        job["match_score"] = round(score, 1)
        job["matched_skills"] = sorted(student_skills & job_tokens)
        ranked.append(job)
    return sorted(ranked, key=lambda x: x["match_score"], reverse=True)
