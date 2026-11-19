from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import streamlit as st

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# =============================================================================
# InternMatch AI — clean single-file Streamlit application
# =============================================================================

st.set_page_config(
    page_title="InternMatch AI | Career Workspace",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
PROFILE_PATH = BASE_DIR / ".internmatch_profile.json"
FEEDBACK_PATH = BASE_DIR / ".internmatch_feedback.json"
APPLICATIONS_PATH = BASE_DIR / ".internmatch_applications.json"
ALERTS_PATH = BASE_DIR / ".internmatch_alerts.json"
JOOBLE_ENDPOINT = "https://in.jooble.org/api/"

# ---------- Design system ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --green: #087443;
        --green-hover: #056238;
        --green-dark: #064E35;
        --green-soft: #E7F4EC;
        --ink: #17231D;
        --muted: #66736B;
        --line: #E3E9E5;
        --surface: #FFFFFF;
        --canvas: #F6F8F6;
    }

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }
    .stApp { background: var(--canvas); }
    [data-testid="stHeader"] { background: rgba(246,248,246,.94); }
    .main .block-container {
        max-width: 1440px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #10291F;
        border-right: 1px solid #214333;
    }
    section[data-testid="stSidebar"] > div { background: #10291F; }
    section[data-testid="stSidebar"] * { color: #EAF3ED; }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label { color: #D4E3D9 !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {
        background: #FFFFFF !important;
        color: #17231D !important;
        border-color: #D7E1DA !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] * { color: #17231D !important; }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        color:#EAF3ED !important; border-radius:9px; padding:8px 10px;
        margin:3px 0; background:transparent; transition:background .15s ease;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background:#087443 !important; color:#FFFFFF !important;
    }

    /* Brand and headings */
    .brand-lockup { display:flex; align-items:center; gap:12px; margin: 0 0 1.5rem 0; }
    section[data-testid="stSidebar"] .brand-mark {
        background:#DDF3E5 !important; color:#087443 !important;
        -webkit-text-fill-color:#087443 !important;
    }
    .brand-mark {
        width:42px; height:42px; display:flex; align-items:center; justify-content:center;
        border-radius:12px; background:#DDF3E5; color:#075D38; font-size:21px; font-weight:800;
    }
    .brand-name { font-family:'Manrope',sans-serif; font-size:18px; font-weight:800; color:#FFFFFF; line-height:1.15; }
    .brand-sub { font-size:11px; color:#AFC8B8; letter-spacing:.08em; text-transform:uppercase; margin-top:3px; }
    .eyebrow { color:#087443; text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:700; margin-bottom:.45rem; }
    .hero-title { font-family:'Manrope',sans-serif; font-size:2.15rem; line-height:1.15; letter-spacing:-.04em; font-weight:800; color:#14251B; margin:0; }
    .hero-copy { color:#64736A; font-size:1rem; margin-top:.55rem; max-width:760px; }
    .section-title { font-family:'Manrope',sans-serif; color:#172A20; font-size:1.2rem; font-weight:800; letter-spacing:-.02em; }
    .muted { color:#6B786F; }

    /* Cards */
    .surface-card {
        background:#FFFFFF; border:1px solid #E1E8E3; border-radius:16px;
        padding:20px 22px; box-shadow:0 5px 18px rgba(18,45,29,.045); margin-bottom:12px;
    }
    .job-title { color:#17231D; font-family:'Manrope',sans-serif; font-weight:800; font-size:1.13rem; margin:0 0 5px 0; }
    .job-meta { color:#68766D; font-size:.88rem; margin-bottom:12px; }
    .job-description { color:#45534A; font-size:.92rem; line-height:1.65; }
    .match-panel {
        background:#075D38; border:1px solid #075D38; color:#FFFFFF;
        border-radius:13px; padding:13px 15px; text-align:center;
    }
    .match-panel .match-label { color:#D7F2E1; font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; font-weight:700; }
    .match-panel .match-value { color:#FFFFFF; font-family:'Manrope',sans-serif; font-size:1.75rem; line-height:1.2; font-weight:800; }
    .status-note { color:#4B6254; background:#F1F6F2; border:1px solid #E1EAE3; border-radius:9px; padding:8px 11px; font-size:.82rem; }
    .stat-card { background:#FFFFFF; border:1px solid #E1E8E3; border-radius:13px; padding:15px 17px; }
    .stat-label { color:#758178; text-transform:uppercase; letter-spacing:.09em; font-size:.68rem; font-weight:700; }
    .stat-value { color:#173323; font-family:'Manrope',sans-serif; font-weight:800; font-size:1.55rem; margin-top:3px; }

    /* Inputs: always white with readable dark text */
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="select"] > div {
        background:#FFFFFF !important; border:1px solid #CBD6CE !important; border-radius:9px !important;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span {
        color:#17231D !important; -webkit-text-fill-color:#17231D !important;
        background:#FFFFFF !important; caret-color:#087443 !important;
    }
    div[data-baseweb="input"] input::placeholder,
    div[data-baseweb="textarea"] textarea::placeholder { color:#87938B !important; -webkit-text-fill-color:#87938B !important; }
    [data-testid="stTextInput"] label, [data-testid="stTextArea"] label,
    [data-testid="stSelectbox"] label, [data-testid="stMultiSelect"] label,
    [data-testid="stNumberInput"] label, [data-testid="stSlider"] label,
    [data-testid="stDateInput"] label { color:#34443A !important; font-weight:600 !important; }

    /* Strong overrides for Streamlit text fields across BaseWeb versions. */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input,
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="select"] input {
        background-color:#FFFFFF !important; color:#17231D !important;
        -webkit-text-fill-color:#17231D !important; caret-color:#087443 !important;
        opacity:1 !important;
    }
    [data-testid="stTextArea"] [data-baseweb="textarea"],
    [data-testid="stTextInput"] [data-baseweb="input"],
    [data-testid="stNumberInput"] [data-baseweb="input"] {
        background-color:#FFFFFF !important;
    }
    [data-testid="stTextArea"] textarea::placeholder,
    [data-testid="stTextInput"] input::placeholder {
        color:#87938B !important; -webkit-text-fill-color:#87938B !important;
        opacity:1 !important;
    }
    /* Resume preview: Streamlit st.text() renders a preformatted block. */
    .resume-preview {
        background:#FFFFFF !important; color:#17231D !important;
        border:1px solid #E1E8E3 !important; border-radius:12px !important;
        padding:20px 22px !important; white-space:pre-wrap !important;
        overflow-wrap:anywhere !important; line-height:1.65 !important;
        font-family:'DM Sans',sans-serif !important; font-size:.94rem !important;
        opacity:1 !important;
    }

    /* Sidebar navigation: keep every label and icon on the same left edge */
    section[data-testid="stSidebar"] [data-testid="stButton"] > button {
        justify-content:flex-start !important;
        text-align:left !important;
        padding-left:1rem !important;
        gap:.65rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] > button > div {
        justify-content:flex-start !important;
        width:100% !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] [data-testid="stMarkdownContainer"] {
        text-align:left !important;
        width:100% !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] [data-testid="stMarkdownContainer"] p {
        text-align:left !important;
        margin:0 !important;
    }

    /* Every clickable button: saturated green + bright white */
    .stButton > button, .stDownloadButton > button, .stLinkButton > a,
    .stFormSubmitButton > button, section[data-testid="stSidebar"] .stButton > button {
        background:#087443 !important; color:#FFFFFF !important;
        border:1px solid #087443 !important; border-radius:9px !important;
        font-weight:700 !important; min-height:42px !important;
        box-shadow:0 2px 5px rgba(8,116,67,.12) !important;
        transition:background .15s ease, transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button *, .stDownloadButton > button *, .stLinkButton > a *,
    .stFormSubmitButton > button *, section[data-testid="stSidebar"] .stButton > button * { color:#FFFFFF !important; fill:#FFFFFF !important; }
    .stButton > button:hover, .stDownloadButton > button:hover, .stLinkButton > a:hover,
    .stFormSubmitButton > button:hover, section[data-testid="stSidebar"] .stButton > button:hover {
        background:#056238 !important; border-color:#056238 !important;
        color:#FFFFFF !important; transform:translateY(-1px);
        box-shadow:0 5px 12px rgba(8,116,67,.2) !important;
    }
    .stButton > button:focus, .stDownloadButton > button:focus { outline:3px solid #B8E4C8 !important; outline-offset:2px !important; }

    /* Sidebar group headers: green while collapsed, transparent while open */
    section[data-testid="stSidebar"] [data-testid="stExpander"] details:not([open]) > summary {
        background:#087443 !important;
        color:#FFFFFF !important;
        border:1px solid #087443 !important;
        border-radius:9px !important;
        font-weight:700 !important;
        box-shadow:0 2px 5px rgba(8,116,67,.12) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] details:not([open]) > summary * {
        color:#FFFFFF !important;
        fill:#FFFFFF !important;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] details:not([open]) > summary:hover {
        background:#056238 !important;
        border-color:#056238 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary {
        background:transparent !important;
        color:#FFFFFF !important;
        border-color:rgba(255,255,255,.22) !important;
        box-shadow:none !important;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary * {
        color:#FFFFFF !important;
        fill:#FFFFFF !important;
    }

    /* Progress and tabs */
    [data-testid="stProgressBar"] > div > div { background:#087443 !important; }
    button[data-baseweb="tab"] { color:#65736A !important; font-weight:700 !important; }

    /* Compact authentication selector: charcoal text, no oversized buttons. */
    .st-key-auth_mode_selector [data-testid="stRadio"] > div {
        gap:1.4rem !important;
    }
    .st-key-auth_mode_selector [data-testid="stRadio"] label {
        color:#292929 !important;
        font-weight:700 !important;
        opacity:1 !important;
    }
    .st-key-auth_mode_selector [data-testid="stRadio"] label p {
        color:#292929 !important;
        -webkit-text-fill-color:#292929 !important;
        opacity:1 !important;
    }
    .st-key-auth_mode_selector [data-testid="stRadio"] input:checked + div {
        background-color:#292929 !important;
        border-color:#292929 !important;
    }
    .st-key-auth_mode_selector [data-testid="stRadio"] input:checked + div svg {
        color:#292929 !important;
        fill:#292929 !important;
    }
    hr { border-color:#E2E9E4 !important; }
    [data-testid="stAlert"] { border-radius:10px; }
    footer { visibility:hidden; }
    
/* Build Profile: dark input fields to match dropdowns */
[data-testid="stForm"] [data-testid="stTextInput"] input,
[data-testid="stForm"] [data-testid="stTextArea"] textarea,
[data-testid="stForm"] [data-testid="stNumberInput"] input,
[data-testid="stForm"] div[data-baseweb="input"] input,
[data-testid="stForm"] div[data-baseweb="textarea"] textarea {
    background-color: #292929 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid #444444 !important;
    border-radius: 9px !important;
    caret-color: #FFFFFF !important;
}

/* Placeholder text */
[data-testid="stForm"] input::placeholder,
[data-testid="stForm"] textarea::placeholder {
    color: #BBBBBB !important;
    -webkit-text-fill-color: #BBBBBB !important;
    opacity: 1 !important;
}

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Supabase account + cloud storage ----------
SUPABASE_READY = False
SUPABASE_CLIENT = None
SUPABASE_IMPORT_ERROR = ""
try:
    from supabase import create_client
    SUPABASE_URL = str(st.secrets.get("SUPABASE_URL", "")).strip()
    SUPABASE_ANON_KEY = str(st.secrets.get("SUPABASE_ANON_KEY", "")).strip()
    if SUPABASE_URL and SUPABASE_ANON_KEY:
        SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        SUPABASE_READY = True
except Exception as exc:
    SUPABASE_IMPORT_ERROR = str(exc)

if not SUPABASE_READY:
    st.error("Supabase is not configured. Add SUPABASE_URL and SUPABASE_ANON_KEY to your Streamlit secrets, and install the supabase package.")
    if SUPABASE_IMPORT_ERROR:
        st.caption(f"Setup detail: {SUPABASE_IMPORT_ERROR}")
    st.stop()

def _set_supabase_session():
    access = st.session_state.get("_sb_access_token")
    refresh = st.session_state.get("_sb_refresh_token")
    if access and refresh:
        try:
            SUPABASE_CLIENT.auth.set_session(access, refresh)
            return True
        except Exception:
            try:
                refreshed = SUPABASE_CLIENT.auth.refresh_session(refresh)
                sess = getattr(refreshed, "session", None)
                if sess:
                    st.session_state["_sb_access_token"] = sess.access_token
                    st.session_state["_sb_refresh_token"] = sess.refresh_token
                    SUPABASE_CLIENT.auth.set_session(sess.access_token, sess.refresh_token)
                    return True
            except Exception:
                pass
    return False

def _cloud_load(user_id: str) -> dict:
    result = SUPABASE_CLIENT.table("user_workspace").select("workspace").eq("user_id", user_id).maybe_single().execute()
    row = getattr(result, "data", None)
    if isinstance(row, dict) and isinstance(row.get("workspace"), dict):
        return row["workspace"]
    return {}

def _cloud_save():
    user_id = st.session_state.get("_sb_user_id")
    if not user_id:
        return
    payload = {
        "profile": st.session_state.get("profile", {}),
        "feedback": st.session_state.get("feedback", {}),
        "saved_jobs": st.session_state.get("saved_jobs", []),
        "applications": st.session_state.get("applications", []),
        "internship_alerts": st.session_state.get("internship_alerts", {}),
    }
    SUPABASE_CLIENT.table("user_workspace").upsert(
        {"user_id": user_id, "workspace": payload}, on_conflict="user_id"
    ).execute()

def _reset_workspace_state():
    for key in [
        "profile", "feedback", "saved_jobs", "applications", "internship_alerts",
        "jobs", "selected_job", "resume_job", "interview_job", "skillgap_job",
        "insights_job", "page", "search_total", "search_query", "search_location",
    ]:
        st.session_state.pop(key, None)

def _auth_screen():
    st.markdown('<div class="eyebrow">WELCOME TO INTERNMATCH AI</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Your career workspace, saved securely.</h1>', unsafe_allow_html=True)
    st.markdown('<div class="hero-copy">Create an account or log in to access your profile, saved internships and application tracker from any device.</div>', unsafe_allow_html=True)

    # Compact horizontal selector instead of large buttons or low-contrast tabs.
    with st.container(key="auth_mode_selector"):
        auth_mode = st.radio(
            "Account access",
            ["Log in", "Create account"],
            horizontal=True,
            label_visibility="collapsed",
            key="_auth_mode_choice",
        )

    if auth_mode == "Log in":
        with st.form("supabase_login"):
            email = st.text_input("Email address", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", use_container_width=True)
        if submitted:
            try:
                result = SUPABASE_CLIENT.auth.sign_in_with_password({"email": email.strip(), "password": password})
                sess = getattr(result, "session", None)
                user = getattr(result, "user", None)
                if sess and user:
                    _reset_workspace_state()
                    st.session_state["_sb_access_token"] = sess.access_token
                    st.session_state["_sb_refresh_token"] = sess.refresh_token
                    st.session_state["_sb_user_id"] = user.id
                    st.session_state["_sb_email"] = user.email or email.strip()
                    st.rerun()
                st.error("Login did not return an active session. Check your email confirmation settings.")
            except Exception as exc:
                st.error(f"Could not log in: {exc}")
    else:
        with st.form("supabase_signup"):
            new_email = st.text_input("Email address", key="signup_email")
            new_password = st.text_input("Create a password (at least 8 characters)", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")
            created = st.form_submit_button("Create account", use_container_width=True)
        if created:
            if len(new_password) < 8:
                st.error("Please use a password with at least 8 characters.")
            elif new_password != confirm_password:
                st.error("The passwords do not match.")
            else:
                try:
                    result = SUPABASE_CLIENT.auth.sign_up({"email": new_email.strip(), "password": new_password})
                    sess = getattr(result, "session", None)
                    user = getattr(result, "user", None)
                    if sess and user:
                        _reset_workspace_state()
                        st.session_state["_sb_access_token"] = sess.access_token
                        st.session_state["_sb_refresh_token"] = sess.refresh_token
                        st.session_state["_sb_user_id"] = user.id
                        st.session_state["_sb_email"] = user.email or new_email.strip()
                        st.rerun()
                    st.success("Account created. Check your email for the confirmation link, then return here and log in.")
                except Exception as exc:
                    st.error(f"Could not create account: {exc}")

def _logout():
    try:
        SUPABASE_CLIENT.auth.sign_out()
    except Exception:
        pass
    for key in ["_sb_access_token", "_sb_refresh_token", "_sb_user_id", "_sb_email", "_cloud_workspace", "_cloud_workspace_loaded"]:
        st.session_state.pop(key, None)
    _reset_workspace_state()
    st.rerun()

_has_session = _set_supabase_session()
if not _has_session or not st.session_state.get("_sb_user_id"):
    _auth_screen()
    st.stop()

try:
    # Confirm the current token still belongs to an active Supabase user.
    _cloud_workspace = _cloud_load(st.session_state["_sb_user_id"])
    st.session_state["_cloud_workspace"] = _cloud_workspace
except Exception as exc:
    st.error(f"Could not load your cloud workspace: {exc}")
    st.info("Check that the user_workspace table and its Row Level Security policies have been created in Supabase.")
    st.stop()


# ---------- Persistence ----------
def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def write_json(path: Path, value: Any) -> None:
    # Keep the existing app's save calls, but route their data to the signed-in
    # user's private Supabase workspace instead of relying on ephemeral files.
    key_by_file = {
        ".internmatch_profile.json": "profile",
        ".internmatch_feedback.json": "feedback",
        ".internmatch_saved_jobs.json": "saved_jobs",
        ".internmatch_applications.json": "applications",
        ".internmatch_alerts.json": "internship_alerts",
    }
    cloud_key = key_by_file.get(path.name)
    if cloud_key and st.session_state.get("_sb_user_id"):
        st.session_state.setdefault("_cloud_workspace", {})[cloud_key] = value
        try:
            _cloud_save()
        except Exception as exc:
            st.error(f"Could not save to your Supabase account: {exc}")
        return
    try:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        st.warning(f"Could not save local data: {exc}")


DEFAULT_PROFILE = {
    "full_name": "",
    "email": "",
    "phone": "",
    "degree": "B.Tech",
    "branch": "Computer Science",
    "college": "",
    "graduation_year": 2027,
    "skills": "Python, Machine Learning, Pandas, NumPy, SQL, TensorFlow",
    "interests": "Artificial Intelligence, Data Science, Natural Language Processing",
    "experience_level": "Student / Fresher",
    "preferred_location": "Gurugram",
    "preferred_mode": "Any",
    "projects": "",
    "experience": "",
    "certifications": "",
    "achievements": "",
    "additional_info": "",
}

if "profile" not in st.session_state:
    cloud_data = st.session_state.get("_cloud_workspace", {})
    stored_profile = cloud_data.get("profile") or read_json(PROFILE_PATH, {})
    st.session_state.profile = {**DEFAULT_PROFILE, **stored_profile}
if "jobs" not in st.session_state:
    st.session_state.jobs = []
if "selected_job" not in st.session_state:
    st.session_state.selected_job = None
if "resume_job" not in st.session_state:
    st.session_state.resume_job = None
if "feedback" not in st.session_state:
    st.session_state.feedback = cloud_data.get("feedback", read_json(FEEDBACK_PATH, {}))
if "saved_jobs" not in st.session_state:
    st.session_state.saved_jobs = cloud_data.get("saved_jobs", read_json(BASE_DIR / ".internmatch_saved_jobs.json", []))
if "applications" not in st.session_state:
    st.session_state.applications = cloud_data.get("applications", read_json(APPLICATIONS_PATH, []))
if "internship_alerts" not in st.session_state:
    st.session_state.internship_alerts = cloud_data.get("internship_alerts", read_json(ALERTS_PATH, {"keywords": "", "location": "", "known_job_keys": [], "new_jobs": [], "last_checked": ""}))
if "interview_job" not in st.session_state:
    st.session_state.interview_job = None
if "skillgap_job" not in st.session_state:
    st.session_state.skillgap_job = None
if "insights_job" not in st.session_state:
    st.session_state.insights_job = None
if "page" not in st.session_state:
    st.session_state.page = "Find Internships"
if "search_error" not in st.session_state:
    st.session_state.search_error = ""
if "search_total" not in st.session_state:
    st.session_state.search_total = 0
if "alert_check_message" not in st.session_state:
    st.session_state.alert_check_message = ""
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "search_location" not in st.session_state:
    st.session_state.search_location = st.session_state.profile.get("preferred_location", "Gurugram")


def get_api_key() -> str:
    candidates = [os.getenv("JOOBLE_API_KEY", ""), os.getenv("JOOBLE_KEY", "")]
    try:
        candidates.extend([
            st.secrets.get("JOOBLE_API_KEY", ""),
            st.secrets.get("jooble_api_key", ""),
            st.secrets.get("JOOBLE_KEY", ""),
        ])
        jooble_section = st.secrets.get("jooble", {})
        if hasattr(jooble_section, "get"):
            candidates.append(jooble_section.get("api_key", ""))
    except Exception:
        pass
    # Also support a project-root secrets.toml, in case that is where the
    # existing local project already stores its key. Never display its value.
    for secrets_path in [BASE_DIR / ".streamlit" / "secrets.toml", BASE_DIR / "secrets.toml"]:
        try:
            if secrets_path.exists():
                import tomllib
                config = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
                candidates.extend([config.get("JOOBLE_API_KEY", ""), config.get("jooble_api_key", ""), config.get("JOOBLE_KEY", "")])
                section = config.get("jooble", {})
                if isinstance(section, dict):
                    candidates.append(section.get("api_key", ""))
        except Exception:
            continue
    return next((str(x).strip() for x in candidates if str(x).strip()), "")


def make_job_key(job: dict[str, Any]) -> str:
    raw = job.get("id") or job.get("link") or job.get("url") or f"{job.get('title','')}_{job.get('company','')}"
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(raw))[:110] or "job_unknown"




def clean_html_text(value: Any) -> str:
    """Clean and format job descriptions from Jooble."""
    text = str(value or "")

    # Decode HTML entities, including double-encoded content.
    for _ in range(3):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded

    # Preserve paragraph and line breaks.
    text = re.sub(
        r"(?i)<\s*(br|/p|/div|/li|/h[1-6])\s*/?\s*>",
        "\n", text
    )
    text = re.sub(
        r"(?i)<\s*(p|div|li|h[1-6])\b[^>]*>",
        "\n", text
    )

    # Remove HTML without splitting words.
    text = re.sub(r"<[^>]*>", "", text)
    text = html.unescape(text).replace("\xa0", " ")

    # Remove invisible formatting characters.
    text = re.sub(r"[\u200e\u200f\u202a-\u202e\ufeff]", "", text)

    # Remove unwanted Summary headings at the beginning.
    text = re.sub(
        r"(?i)^\s*[.…\s-]*(?:job\s+)?summary\s*[:\-–—]?\s*",
        "", text
    )

    # Replace repeated ellipses with clean sentence punctuation.
    text = re.sub(r"(?:\.{2,}|…+)", ". ", text)

    # Clean spacing and punctuation.
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip(" .\n")




def normalize_job(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(raw.get("id", "") or ""),
        "title": str(raw.get("title", "Internship opportunity") or "Internship opportunity"),
        "company": str(raw.get("company", "Company not provided") or "Company not provided"),
        "location": str(raw.get("location", "Location not specified") or "Location not specified"),
        "description": clean_html_text(raw.get("snippet") or raw.get("description") or "Description not provided by the listing."),
        "url": str(raw.get("link", raw.get("url", "")) or ""),
        "salary": str(raw.get("salary", "") or ""),
        "type": str(raw.get("type", "") or ""),
        "source": str(raw.get("source", "Jooble") or "Jooble"),
        "updated": str(raw.get("updated", "") or ""),
    }


def jooble_search(keywords: str, location: str, api_key: str, count: int = 30) -> tuple[list[dict[str, Any]], int]:
    if not api_key:
        raise RuntimeError(
            "Jooble API key not found. Add JOOBLE_API_KEY = \"your_key\" to .streamlit/secrets.toml."
        )
    url = JOOBLE_ENDPOINT + api_key
    payload = {
        "keywords": keywords,
        "location": location,
        "page": "1",
        "ResultOnPage": int(count),
        "companysearch": "false",
    }
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=25,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach Jooble: {exc}") from exc

    if response.status_code == 403:
        raise RuntimeError(
            "Jooble returned 403 Access Denied. The API key may be invalid, or it may belong to a different country domain. "
            "For India, use a key issued through in.jooble.org/api/about."
        )
    if response.status_code == 429:
        raise RuntimeError("Jooble request limit reached (HTTP 429). Please try again after the daily limit resets.")
    if response.status_code == 404:
        raise RuntimeError("Jooble endpoint not found (HTTP 404). Check that the India API key is being used with in.jooble.org.")
    if not response.ok:
        detail = response.text[:350].strip()
        raise RuntimeError(f"Jooble request failed (HTTP {response.status_code}). {detail}")
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Jooble returned a response that was not valid JSON.") from exc
    raw_jobs = data.get("jobs", []) if isinstance(data, dict) else []
    if not isinstance(raw_jobs, list):
        raw_jobs = []
    jobs = [normalize_job(x) for x in raw_jobs if isinstance(x, dict)]
    return jobs, int(data.get("totalCount", len(jobs)) or len(jobs))


# ---------- Recommendation scoring ----------
def tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9+#.]+", str(value).lower()))


def phrase_skill_score(student_skills: str, listing_text: str) -> float:
    phrases = [x.strip().lower() for x in re.split(r"[,;\n]", student_skills) if x.strip()]
    if not phrases:
        return 0.0
    text = listing_text.lower()
    exact = sum(1 for phrase in phrases if phrase in text) / len(phrases)
    student_tokens = tokens(student_skills)
    listing_tokens = tokens(listing_text)
    overlap = len(student_tokens & listing_tokens) / max(1, len(student_tokens))
    return min(1.0, 0.7 * exact + 0.3 * overlap)


def score_jobs(jobs: list[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
    if not jobs:
        return []
    profile_text = " ".join([
        profile.get("skills", ""), profile.get("interests", ""),
        profile.get("degree", ""), profile.get("branch", ""),
        profile.get("experience_level", ""), profile.get("projects", ""),
    ])
    docs = [f"{j['title']} {j['company']} {j['description']} {j['location']} {j['type']}" for j in jobs]
    text_scores = [0.0] * len(jobs)
    interest_scores = [0.0] * len(jobs)
    if SKLEARN_AVAILABLE and any(x.strip() for x in docs):
        try:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=6000)
            matrix = vectorizer.fit_transform(docs + [profile_text or "student internship"])
            text_scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten().tolist()
            interest_vector = vectorizer.transform([profile.get("interests", "") or "student"])
            interest_scores = cosine_similarity(interest_vector, matrix[:-1]).flatten().tolist()
        except ValueError:
            pass
    skills = profile.get("skills", "")
    interests = profile.get("interests", "")
    education = f"{profile.get('degree','')} {profile.get('branch','')}".strip().lower()
    preferred_location = profile.get("preferred_location", "").strip().lower()
    preferred_mode = profile.get("preferred_mode", "Any")

    ranked = []
    for index, job in enumerate(jobs):
        full_text = docs[index]
        skill_score = phrase_skill_score(skills, full_text)
        interest_score = interest_scores[index] if index < len(interest_scores) else 0.0
        education_score = 0.5 if not education else (1.0 if any(t in full_text.lower() for t in tokens(education)) else 0.4)
        location_text = job.get("location", "").lower()
        location_score = 1.0 if preferred_location and preferred_location in location_text else (0.5 if not preferred_location else 0.0)
        mode_text = f"{job.get('type','')} {job.get('description','')}".lower()
        mode_score = 1.0 if preferred_mode == "Any" or preferred_mode.lower() in mode_text else 0.4
        location_mode_score = 0.6 * location_score + 0.4 * mode_score
        text_score = text_scores[index] if index < len(text_scores) else 0.0
        # Same transparent prototype weights used in the project's synopsis.
        match = 0.45 * text_score + 0.30 * skill_score + 0.15 * interest_score + 0.05 * education_score + 0.05 * location_mode_score
        item = dict(job)
        item.update({
            "text_score": float(text_score),
            "skill_score": float(skill_score),
            "interest_score": float(interest_score),
            "education_score": float(education_score),
            "location_score": float(location_mode_score),
            "match_score": float(match),
        })
        ranked.append(item)
    return sorted(ranked, key=lambda x: x["match_score"], reverse=True)


def persist_feedback() -> None:
    write_json(FEEDBACK_PATH, st.session_state.feedback)
    write_json(BASE_DIR / ".internmatch_saved_jobs.json", st.session_state.saved_jobs)


def safe_link(url: str) -> bool:
    try:
        return urlparse(url).scheme in {"http", "https"}
    except Exception:
        return False



def generate_interview_questions(job: dict[str, Any], profile: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Create role-specific interview practice questions from the listing and student profile."""
    title = str(job.get("title", "internship"))
    company = str(job.get("company", "the company"))
    description = str(job.get("description", ""))
    profile_skills = str(profile.get("skills", ""))
    projects = str(profile.get("projects", ""))
    context = f"{title} {description} {profile_skills} {projects}".lower()

    skill_questions = {
        "python": ("Python", "How have you used Python in a project? Walk me through the approach, libraries and one challenge you faced."),
        "sql": ("SQL", "How would you use SQL to find the top five records in a table, and when would you use GROUP BY?"),
        "machine learning": ("Machine Learning", "Explain the end-to-end machine-learning workflow you would follow for a new prediction problem."),
        "deep learning": ("Deep Learning", "What is the difference between a CNN and a traditional fully connected neural network, and when would you use a CNN?"),
        "nlp": ("NLP", "How would you prepare text data for an NLP task, and how would you evaluate the model?"),
        "natural language processing": ("NLP", "How would you prepare text data for an NLP task, and how would you evaluate the model?"),
        "tensorflow": ("TensorFlow", "Describe how you would build, train and evaluate a model using TensorFlow."),
        "pytorch": ("PyTorch", "Explain the basic training loop in PyTorch, including loss calculation, backpropagation and optimizer steps."),
        "data analysis": ("Data Analysis", "How would you handle missing values, outliers and inconsistent data before analysing a dataset?"),
        "pandas": ("Pandas", "How would you use Pandas to clean a dataset and summarise important trends?"),
        "numpy": ("NumPy", "What are NumPy arrays, and why are they useful for numerical computing?"),
        "statistics": ("Statistics", "Explain the difference between correlation and causation with an example."),
        "java": ("Java", "Explain the four pillars of object-oriented programming and give a practical example of one."),
        "javascript": ("JavaScript", "What is the difference between let, const and var, and when would you use each?"),
        "react": ("React", "What are React components and hooks? Describe how you have used them."),
        "html": ("Web Development", "How do HTML, CSS and JavaScript work together to create a web page?"),
        "css": ("Web Development", "How do HTML, CSS and JavaScript work together to create a web page?"),
        "flask": ("Flask", "How would you create an API endpoint in Flask and connect it to a front end?"),
        "api": ("APIs", "What happens when an application sends an API request, and how would you handle an unsuccessful response?"),
        "aws": ("Cloud", "Which cloud services have you explored, and how would you decide where to deploy a small application?"),
        "power bi": ("Power BI", "How would you turn a raw dataset into a useful dashboard for a business stakeholder?"),
        "excel": ("Excel", "Which Excel features would you use to clean, analyse and summarise a large dataset?"),
        "git": ("Git", "How do you use Git to manage changes when working on a software project?"),
        "computer vision": ("Computer Vision", "How would you prepare image data and evaluate a model that detects defective products?"),
        "opencv": ("Computer Vision", "What image-processing steps might be useful before passing an image to a computer-vision model?"),
    }

    technical = []
    seen_topics = set()
    for keyword, (label, question) in skill_questions.items():
        if keyword in context and label.lower() not in seen_topics:
            technical.append({
                "question": question,
                "why": f"This checks your understanding of {label}, which appears in the internship or your profile.",
                "tip": f"Define the concept, explain how it works, and connect it to a project or practical example."
            })
            seen_topics.add(label.lower())
        if len(technical) >= 7:
            break

    technical.insert(0, {
        "question": f"What do you understand about the responsibilities of a {title} intern, and how would you approach your first assignment?",
        "why": "This connects your understanding directly to the role you selected.",
        "tip": "Refer to responsibilities and skills mentioned in the listing. Explain the steps you would take rather than giving a generic answer."
    })
    technical.append({
        "question": f"Looking at the requirements for this {title} internship, which area would be the most challenging for you and how would you get up to speed?",
        "why": "Interviewers often check how honestly you assess your current skills and learn new ones.",
        "tip": "Choose a genuine learning area and describe a practical plan: documentation, a small project, practice and feedback."
    })

    hr = [
        {
            "question": f"Tell me about yourself and why you are interested in this {title} internship at {company}.",
            "tip": "Give a short introduction, mention your relevant education and projects, then connect your interests to this role."
        },
        {
            "question": "Walk me through a project you have worked on that is most relevant to this internship.",
            "tip": "Explain the problem, your individual contribution, tools used, one challenge and the result."
        },
        {
            "question": "Tell me about a time you faced a technical challenge. How did you solve it?",
            "tip": "Use a real example. Structure it as situation, task, action and result."
        },
        {
            "question": "How do you manage deadlines when you have multiple tasks to complete?",
            "tip": "Describe how you prioritise, break work into smaller tasks and communicate if something is delayed."
        },
        {
            "question": "What is one skill you are currently working to improve?",
            "tip": "Choose a genuine skill and explain what you are doing to improve it."
        },
        {
            "question": "What would you like to learn or contribute during this internship?",
            "tip": "Mention one or two role-related skills you want to develop and how you could contribute as an intern."
        },
    ]

    return {"Technical / Role-based": technical, "HR / Behavioural": hr}



def analyse_skill_gap(job: dict[str, Any], profile: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Compare skills mentioned in a listing with skills in the student's profile."""
    skill_guides = {
        "python": ("Python", "Practise variables, functions, data structures, file handling and writing small scripts.", "Build a small project that reads data, processes it and produces a useful result."),
        "sql": ("SQL", "Learn SELECT, filtering, JOINs, GROUP BY, subqueries and aggregate functions.", "Create a small database and answer 10 business questions using SQL queries."),
        "machine learning": ("Machine Learning", "Study supervised and unsupervised learning, train/test splits, overfitting and evaluation metrics.", "Train and compare two models on a public dataset; explain why one performs better."),
        "deep learning": ("Deep Learning", "Learn neural-network basics, activation functions, loss, backpropagation and regularisation.", "Train a small neural network and document its training and validation results."),
        "natural language processing": ("Natural Language Processing", "Study text cleaning, tokenisation, embeddings, classification and evaluation.", "Build a text-classification or sentiment-analysis mini project."),
        "nlp": ("NLP", "Study text cleaning, tokenisation, embeddings, classification and evaluation.", "Build a text-classification or sentiment-analysis mini project."),
        "tensorflow": ("TensorFlow", "Learn tensors, model layers, compile/fit workflows and model evaluation.", "Train a simple classification model and save the trained model."),
        "pytorch": ("PyTorch", "Learn tensors, datasets, DataLoaders, model classes, loss functions and optimisers.", "Implement a small training loop for an image or tabular dataset."),
        "pandas": ("Pandas", "Practise loading files, filtering rows, handling missing values, grouping and merging data.", "Analyse a dataset and create a short notebook with clear findings."),
        "numpy": ("NumPy", "Learn arrays, shapes, indexing, broadcasting and vectorised operations.", "Solve a set of numerical tasks using NumPy arrays instead of Python loops."),
        "data analysis": ("Data Analysis", "Practise data cleaning, exploratory analysis, visualisation and communicating insights.", "Analyse a real dataset and present three evidence-based findings."),
        "statistics": ("Statistics", "Revise distributions, mean/median, variance, probability, correlation and hypothesis testing.", "Use a dataset to explain its distribution and test one simple hypothesis."),
        "java": ("Java", "Revise OOP, collections, exception handling, strings and basic problem solving.", "Build a small object-oriented application and explain its class design."),
        "javascript": ("JavaScript", "Learn variables, functions, arrays, objects, DOM basics, promises and async/await.", "Create a small interactive web page that fetches and displays data."),
        "react": ("React", "Learn components, props, state, hooks and basic API integration.", "Build a small React interface with reusable components and API data."),
        "html": ("HTML", "Learn semantic elements, forms, page structure and accessibility basics.", "Create a clean multi-section webpage with a working form."),
        "css": ("CSS", "Practise box model, flexbox, grid, responsive layouts and reusable styling.", "Make a responsive page that works on both desktop and mobile."),
        "flask": ("Flask", "Learn routes, request/response handling, templates and basic API development.", "Create a small Flask API and connect it to a simple front end."),
        "api": ("API Development", "Understand HTTP methods, status codes, JSON, authentication basics and error handling.", "Connect an application to a public API and handle loading and error states."),
        "aws": ("AWS / Cloud", "Learn core cloud concepts, IAM basics, compute, storage and deployment fundamentals.", "Deploy a small project and document the services and configuration used."),
        "power bi": ("Power BI", "Learn data loading, Power Query, relationships, measures and dashboard design.", "Build a dashboard from a public dataset and explain its key metrics."),
        "excel": ("Excel", "Practise formulas, lookups, pivot tables, charts and data cleaning.", "Create a small analysis workbook with a pivot table and dashboard."),
        "git": ("Git", "Learn repositories, commits, branches, merges and resolving simple conflicts.", "Put a project on GitHub with a clear README and meaningful commit history."),
        "computer vision": ("Computer Vision", "Study image preprocessing, CNN basics, augmentation and classification metrics.", "Build an image-classification prototype and inspect its errors."),
        "opencv": ("OpenCV", "Practise image loading, resizing, colour spaces, filtering and contour operations.", "Create a mini image-processing application using OpenCV."),
        "docker": ("Docker", "Learn images, containers, Dockerfiles, ports and environment variables.", "Containerise a small application and run it locally."),
        "communication": ("Communication", "Practise explaining technical work clearly, asking clarifying questions and writing concise updates.", "Prepare a two-minute explanation of one project for a non-technical listener."),
        "data structures": ("Data Structures", "Revise arrays, strings, stacks, queues, linked lists, trees and hash maps.", "Solve beginner-level problems and explain time and space complexity."),
    }

    listing_text = " ".join([
        str(job.get("title", "")),
        str(job.get("description", "")),
        str(job.get("type", "")),
    ]).lower()
    student_text = " ".join([
        str(profile.get("skills", "")),
        str(profile.get("projects", "")),
        str(profile.get("experience", "")),
    ]).lower()

    required = []
    for keyword, (label, learn, project) in skill_guides.items():
        pattern = r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])"
        if re.search(pattern, listing_text):
            # Avoid duplicate labels such as NLP and Natural Language Processing.
            if not any(item["skill"] == label for item in required):
                required.append({
                    "skill": label,
                    "learn": learn,
                    "project": project,
                    "status": "Already in profile" if re.search(pattern, student_text) else "Needs attention",
                })

    present = [item for item in required if item["status"] == "Already in profile"]
    gaps = [item for item in required if item["status"] == "Needs attention"]
    return {"matched": present, "gaps": gaps, "all": required}


def make_resume_text(profile: dict[str, Any], job: dict[str, Any] | None) -> str:
    target = job or {}
    name = profile.get("full_name") or "Your Name"
    contact = " | ".join(x for x in [profile.get("email", ""), profile.get("phone", ""), profile.get("preferred_location", "")] if x)
    lines = [name.upper(), contact, "", "PROFESSIONAL SUMMARY",
             f"Final-year {profile.get('degree','')} student in {profile.get('branch','')}, interested in {profile.get('interests','')}. "
             f"Brings practical skills in {profile.get('skills','')} and a strong interest in applying technology to real-world problems."]
    if target:
        lines += ["", "TARGET ROLE", f"{target.get('title','')} — {target.get('company','')}"]
    lines += ["", "EDUCATION", f"{profile.get('degree','')} — {profile.get('branch','')}",
              profile.get("college", ""), f"Expected graduation: {profile.get('graduation_year','')}"]
    if profile.get("skills"):
        lines += ["", "TECHNICAL SKILLS", profile["skills"]]
    if profile.get("projects"):
        lines += ["", "PROJECTS", profile["projects"]]
    if profile.get("experience"):
        lines += ["", "EXPERIENCE", profile["experience"]]
    if profile.get("certifications"):
        lines += ["", "CERTIFICATIONS & COURSES", profile["certifications"]]
    if profile.get("achievements"):
        lines += ["", "ACHIEVEMENTS & ACTIVITIES", profile["achievements"]]
    if profile.get("additional_info"):
        lines += ["", "ADDITIONAL INFORMATION", profile["additional_info"]]
    return "\n".join(str(x) for x in lines if x is not None)


def create_docx_bytes(text: str) -> bytes | None:
    try:
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from io import BytesIO
        doc = Document()
        section = doc.sections[0]
        section.top_margin = Inches(0.55)
        section.bottom_margin = Inches(0.55)
        section.left_margin = Inches(0.7)
        section.right_margin = Inches(0.7)
        normal = doc.styles["Normal"]
        normal.font.name = "Aptos"
        normal.font.size = Pt(9.5)
        normal.font.color.rgb = RGBColor(35, 48, 40)
        paragraphs = text.splitlines()
        for index, line in enumerate(paragraphs):
            if not line.strip():
                continue
            if index == 0:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(line)
                run.bold = True
                run.font.size = Pt(20)
                run.font.color.rgb = RGBColor(7, 91, 54)
            elif line in {"PROFESSIONAL SUMMARY", "TARGET ROLE", "EDUCATION", "TECHNICAL SKILLS", "PROJECTS", "EXPERIENCE", "CERTIFICATIONS & COURSES", "ACHIEVEMENTS & ACTIVITIES", "ADDITIONAL INFORMATION"}:
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(8)
                p.paragraph_format.space_after = Pt(2)
                run = p.add_run(line)
                run.bold = True
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(7, 91, 54)
            else:
                p = doc.add_paragraph(line)
                p.paragraph_format.space_after = Pt(1)
        output = BytesIO()
        doc.save(output)
        return output.getvalue()
    except Exception:
        return None

# ---------- Sidebar navigation ----------
with st.sidebar:
    st.markdown(
        '<div class="brand-lockup"><div class="brand-mark">IM</div><div><div class="brand-name">InternMatch AI</div><div class="brand-sub">Student career workspace</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='color:#AFC8B8;font-size:.72rem;font-weight:700;letter-spacing:.12em;margin-bottom:8px'>WORKSPACE</div>", unsafe_allow_html=True)
    # Keep the primary actions visible and group related workspace tools
    # so the sidebar is easier to scan.
    def sidebar_nav_button(page_name: str) -> None:
        button_label = f"✓  {page_name}" if st.session_state.page == page_name else page_name
        nav_icons = {
            "Build Profile": "person",
            "Find Internships": "search",
            "Saved Internships": "bookmark",
            "Application Tracker": "assignment",
            "Career Insights Dashboard": "insights",
            "AI Interview Prep": "psychology",
            "Skill Gap Analysis": "school",
            "Internship Alerts": "notifications",
            "Your Resume": "description",
        }
        icon_name = nav_icons.get(page_name)
        if st.button(
            button_label,
            key=f"nav_{page_name.lower().replace(' ', '_')}",
            icon=f":material/{icon_name}:" if icon_name else None,
            use_container_width=True,
        ):
            st.session_state.page = page_name
            st.rerun()

    sidebar_nav_button("Build Profile")
    sidebar_nav_button("Find Internships")

    with st.expander(
        "MY ACTIVITY",
        expanded=st.session_state.page in {
            "Saved Internships",
            "Application Tracker",
            "Career Insights Dashboard",
        },
    ):
        sidebar_nav_button("Saved Internships")
        sidebar_nav_button("Application Tracker")
        sidebar_nav_button("Career Insights Dashboard")

    with st.expander(
        "CAREER TOOLKIT",
        expanded=st.session_state.page in {
            "AI Interview Prep",
            "Skill Gap Analysis",
        },
    ):
        sidebar_nav_button("AI Interview Prep")
        sidebar_nav_button("Skill Gap Analysis")

    sidebar_nav_button("Internship Alerts")
    sidebar_nav_button("Your Resume")
    st.markdown("<hr style='border-color:#315342'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#AFC8B8;font-size:.72rem;font-weight:700;letter-spacing:.12em'>PROFILE STATUS</div>", unsafe_allow_html=True)
    profile_complete = bool(st.session_state.profile.get("full_name") and st.session_state.profile.get("skills"))
    st.markdown(
        f"<div style='color:#FFFFFF;font-size:.9rem;margin-top:7px'>{'● Profile ready' if profile_complete else '○ Profile needs details'}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Signed in as {st.session_state.get('_sb_email', 'your account')}. Your workspace is saved to Supabase.")
    if st.button("Log out", key="account_logout", use_container_width=True):
        _logout()
    st.markdown("<hr style='border-color:#315342'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#AFC8B8;font-size:.72rem;font-weight:700;letter-spacing:.12em'>DATA CONNECTION</div>", unsafe_allow_html=True)
    if get_api_key():
        st.markdown("<div style='color:#C9F2D7;font-size:.88rem;margin-top:7px'>● Jooble key detected</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#F6D6A8;font-size:.88rem;margin-top:7px'>○ API key not configured</div>", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<div class="eyebrow">STUDENT CAREER WORKSPACE</div>', unsafe_allow_html=True)
if st.session_state.page == "Build Profile":
    st.markdown('<h1 class="hero-title">Build your profile.</h1>', unsafe_allow_html=True)
    st.markdown('<div class="hero-copy">One strong profile powers more relevant internship matches and tailored application documents.</div>', unsafe_allow_html=True)
elif st.session_state.page == "Find Internships":
    st.markdown('<h1 class="hero-title">Find the internship that fits your next step.</h1>', unsafe_allow_html=True)
    st.markdown('<div class="hero-copy">Search live opportunities, compare your match scores and keep your shortlist in one place.</div>', unsafe_allow_html=True)
elif st.session_state.page == "Saved Internships":
    st.markdown('<h1 class="hero-title">Your saved internships.</h1>', unsafe_allow_html=True)
    st.markdown('<div class="hero-copy">Your shortlist is saved locally on this computer, ready for you to revisit.</div>', unsafe_allow_html=True)
elif st.session_state.page == "Application Tracker":
    st.markdown('<h1 class="hero-title">Track your applications.</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">Keep track of every opportunity, application status and follow-up.</div>',
        unsafe_allow_html=True
    )
elif st.session_state.page == "AI Interview Prep":
    st.markdown('<h1 class="hero-title">Prepare for your interview.</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">Practise technical and HR questions tailored to the internship you selected.</div>',
        unsafe_allow_html=True
    )
elif st.session_state.page == "Skill Gap Analysis":
    st.markdown('<h1 class="hero-title">Build the skills this role needs.</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">Compare an internship’s listed skills with your profile and get a practical learning roadmap.</div>',
        unsafe_allow_html=True
    )
elif st.session_state.page == "Career Insights Dashboard":
    st.markdown('<h1 class="hero-title">Your career insights.</h1>', unsafe_allow_html=True)
    st.markdown('<div class="hero-copy">See patterns across the internships you have found, saved and tracked.</div>', unsafe_allow_html=True)
elif st.session_state.page == "Internship Alerts":
    st.markdown('<h1 class="hero-title">Never miss a new opportunity.</h1>', unsafe_allow_html=True)
    st.markdown('<div class="hero-copy">Save your search preferences and check Jooble for newly listed internships that fit them.</div>', unsafe_allow_html=True)
else:
    st.markdown('<h1 class="hero-title">Your application-ready resume.</h1>', unsafe_allow_html=True)
    st.markdown('<div class="hero-copy">Create a clean, role-focused resume using the details saved in your student profile.</div>', unsafe_allow_html=True)
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# =============================================================================
# BUILD PROFILE
# =============================================================================
if st.session_state.page == "Build Profile":
    p = st.session_state.profile
    with st.form("profile_form", clear_on_submit=False):
        st.markdown('<div class="section-title">Personal details</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            full_name = st.text_input("Full name", value=p.get("full_name", ""), placeholder="Your full name")
            email = st.text_input("Email", value=p.get("email", ""), placeholder="name@example.com")
            phone = st.text_input("Phone (optional)", value=p.get("phone", ""), placeholder="+91 …")
        with c2:
            college = st.text_input("College / University", value=p.get("college", ""), placeholder="University name")
            degree = st.selectbox("Degree", ["B.Tech", "B.E.", "B.Sc.", "M.Tech", "M.Sc.", "Other"], index=["B.Tech", "B.E.", "B.Sc.", "M.Tech", "M.Sc.", "Other"].index(p.get("degree", "B.Tech")) if p.get("degree", "B.Tech") in ["B.Tech", "B.E.", "B.Sc.", "M.Tech", "M.Sc.", "Other"] else 0)
            branch = st.text_input("Degree and branch", value=p.get("branch", "Computer Science"), placeholder="Computer Science / AI")
            graduation_year = st.number_input("Graduation year", min_value=2024, max_value=2035, value=int(p.get("graduation_year", 2027)), step=1)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Skills and career preferences</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            skills = st.text_area("Technical skills", value=p.get("skills", ""), height=100, placeholder="Python, SQL, Machine Learning…")
            interests = st.text_area("Interests / domains", value=p.get("interests", ""), height=100, placeholder="AI, Data Science, NLP…")
            experience_level = st.selectbox("Experience level", ["Student / Fresher", "Beginner", "Intermediate", "Experienced"], index=["Student / Fresher", "Beginner", "Intermediate", "Experienced"].index(p.get("experience_level", "Student / Fresher")) if p.get("experience_level", "Student / Fresher") in ["Student / Fresher", "Beginner", "Intermediate", "Experienced"] else 0)
        with c4:
            preferred_location = st.text_input("Preferred location", value=p.get("preferred_location", "Gurugram"), placeholder="Gurugram, Delhi, Remote…")
            preferred_mode = st.selectbox("Preferred work mode", ["Any", "Remote", "Hybrid", "On-site"], index=["Any", "Remote", "Hybrid", "On-site"].index(p.get("preferred_mode", "Any")) if p.get("preferred_mode", "Any") in ["Any", "Remote", "Hybrid", "On-site"] else 0)
            projects = st.text_area("Projects", value=p.get("projects", ""), height=100, placeholder="Project name — tools used — what you built…")
            experience = st.text_area("Experience / internships (optional)", value=p.get("experience", ""), height=100, placeholder="Role, organisation, dates and contributions…")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Additional information</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            certifications = st.text_area("Certifications and courses (optional)", value=p.get("certifications", ""), height=85)
        with c6:
            achievements = st.text_area("Achievements and activities (optional)", value=p.get("achievements", ""), height=85)
        additional_info = st.text_area("Additional information (optional)", value=p.get("additional_info", ""), height=80)
        submitted = st.form_submit_button("Save my profile", use_container_width=True)

    if submitted:
        new_profile = {
            "full_name": full_name.strip(), "email": email.strip(), "phone": phone.strip(),
            "college": college.strip(), "degree": degree, "branch": branch.strip(),
            "graduation_year": int(graduation_year), "skills": skills.strip(),
            "interests": interests.strip(), "experience_level": experience_level,
            "preferred_location": preferred_location.strip(), "preferred_mode": preferred_mode,
            "projects": projects.strip(), "experience": experience.strip(),
            "certifications": certifications.strip(), "achievements": achievements.strip(),
            "additional_info": additional_info.strip(),
        }
        st.session_state.profile = new_profile
        write_json(PROFILE_PATH, new_profile)
        st.success("Profile saved. Your details will load automatically next time.")

# =============================================================================
# FIND INTERNSHIPS
# =============================================================================
elif st.session_state.page == "Find Internships":
    profile = st.session_state.profile
    metric_cols = st.columns(3)
    metrics = [
        ("LISTINGS FOUND", str(st.session_state.search_total)),
        ("SHORTLISTED", str(len(st.session_state.saved_jobs))),
        ("PROFILE", "Ready" if profile.get("full_name") and profile.get("skills") else "Incomplete"),
    ]
    for col, (label, value) in zip(metric_cols, metrics):
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="section-title">Search opportunities</div>', unsafe_allow_html=True)
        default_terms = ", ".join(x.strip() for x in [profile.get("interests", ""), profile.get("skills", "")] if x.strip())
        default_terms = (default_terms or "AI/ML, Data Science, NLP")[:180]
        s1, s2, s3 = st.columns([2.2, 1.2, .7])
        with s1:
            search_keywords = st.text_input("Search keywords", value=st.session_state.search_query or default_terms, help="Use role titles and skills, for example: AI/ML internship, Data Science, NLP.")
        with s2:
            search_location = st.text_input("Location", value=st.session_state.search_location or profile.get("preferred_location", "Gurugram"))
        with s3:
            result_count = st.selectbox("Results", [10, 20, 30, 40], index=2)
        search_clicked = st.button("Search internships", use_container_width=True)
        st.caption("Live listings are supplied by Jooble. Always verify eligibility, availability and application details on the original listing.")

    if search_clicked:
        st.session_state.search_query = search_keywords.strip()
        st.session_state.search_location = search_location.strip()
        st.session_state.search_error = ""
        if not search_keywords.strip() or not search_location.strip():
            st.session_state.search_error = "Enter both search keywords and a location."
        else:
            with st.spinner("Searching Jooble for current opportunities…"):
                try:
                    api_jobs, total = jooble_search(search_keywords.strip(), search_location.strip(), get_api_key(), result_count)
                    st.session_state.jobs = score_jobs(api_jobs, profile)
                    st.session_state.search_total = total
                    st.session_state.search_error = ""
                    st.session_state.selected_job = None
                except Exception as exc:
                    st.session_state.jobs = []
                    st.session_state.search_total = 0
                    st.session_state.search_error = str(exc)

    if st.session_state.search_error:
        st.error(st.session_state.search_error)
        if "403" in st.session_state.search_error or "Access Denied" in st.session_state.search_error:
            st.markdown("[Check the Jooble India API registration and key](https://in.jooble.org/api/about)")
    if st.session_state.jobs:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Internships matched to your profile</div>', unsafe_allow_html=True)
        st.caption(f"{st.session_state.search_total} listings returned by Jooble. Sorted by your prototype match score.")

        selected = st.session_state.selected_job
        if selected:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('<div class="eyebrow">SELECTED OPPORTUNITY</div>', unsafe_allow_html=True)
                st.markdown(f"## {html.escape(selected.get('title','Opportunity'))}")
                st.markdown(f"**{html.escape(selected.get('company','Company not provided'))}** · {html.escape(selected.get('location',''))}")
                if selected.get("salary") or selected.get("type"):
                    st.caption(" · ".join(x for x in [selected.get("type", ""), selected.get("salary", "")] if x))
                st.write(clean_html_text(selected.get("description", "Description not provided.")))
                st.caption("Always verify mandatory requirements and availability on the original listing.")
                d1, d2 = st.columns(2)
                with d1:
                    if safe_link(selected.get("url", "")):
                        st.link_button("Open original application listing ↗", selected["url"], use_container_width=True)
                    else:
                        st.info("Original listing link was not supplied by Jooble.")
                with d2:
                    if st.button("Create resume for this role", key=f"resume_selected_{make_job_key(selected)}", use_container_width=True):
                        st.session_state.resume_job = dict(selected)
                        st.session_state.page = "Your Resume"
                        st.rerun()
                if st.button("Prepare for interview", key=f"interview_selected_{make_job_key(selected)}", use_container_width=True):
                    st.session_state.interview_job = dict(selected)
                    st.session_state.page = "AI Interview Prep"
                    st.rerun()
                if st.button("Check my skill gaps", key=f"skillgap_selected_{make_job_key(selected)}", use_container_width=True):
                    st.session_state.skillgap_job = dict(selected)
                    st.session_state.page = "Skill Gap Analysis"
                    st.rerun()
                tracker_key = make_job_key(selected)

                if st.button(
                    "＋ Add to Application Tracker",
                    key=f"track_selected_{tracker_key}",
                    use_container_width=True
                ):
                    already_added = any(
                        make_job_key(item.get("job", {})) == tracker_key
                        for item in st.session_state.applications
                    )

                    if not already_added:
                        st.session_state.applications.append({
                            "job": dict(selected),
                            "status": "Interested",
                            "notes": ""
                        })
                        write_json(APPLICATIONS_PATH, st.session_state.applications)
                        st.success("Added to your Application Tracker!")
                    else:
                        st.info("This internship is already in your tracker.")
                if st.button("Close details", key="close_selected_details", use_container_width=True):
                    st.session_state.selected_job = None
                    st.rerun()

        for index, job in enumerate(st.session_state.jobs):
            job_key = make_job_key(job)
            feedback = st.session_state.feedback.get(job_key, "")
            is_saved = any(make_job_key(x) == job_key for x in st.session_state.saved_jobs)
            with st.container(border=True):
                left, right = st.columns([4.4, 1])
                with left:
                    st.markdown(f'<div class="job-title">{html.escape(job.get("title", "Opportunity"))}</div>', unsafe_allow_html=True)
                    meta = " · ".join(x for x in [job.get("company", ""), job.get("location", ""), job.get("type", ""), job.get("salary", "")] if x)
                    st.markdown(f'<div class="job-meta">{html.escape(meta)}</div>', unsafe_allow_html=True)
                    description = clean_html_text(job.get("description", ""))
                    short_description = description[:430] + ("…" if len(description) > 430 else "")
                    st.markdown(f'<div class="job-description">{html.escape(short_description)}</div>', unsafe_allow_html=True)
                    if safe_link(job.get("url", "")):
                        st.markdown(f'<div style="margin-top:9px"><a href="{html.escape(job["url"], quote=True)}" target="_blank" style="color:#087443;font-weight:700">View original listing ↗</a></div>', unsafe_allow_html=True)
                with right:
                    st.markdown(f'<div class="match-panel"><div class="match-label">Profile match</div><div class="match-value">{job.get("match_score",0)*100:.0f}%</div></div>', unsafe_allow_html=True)
                    st.caption("Prototype score")
                    if st.button("View role", key=f"view_role_{job_key}", use_container_width=True):
                        st.session_state.selected_job = dict(job)
                        st.rerun()
                st.progress(min(1.0, max(0.0, float(job.get("match_score", 0)))))
                st.caption(
                    f"Skill match {job.get('skill_score',0)*100:.0f}%  ·  "
                    f"Text similarity {job.get('text_score',0)*100:.0f}%  ·  "
                    f"Education fit {job.get('education_score',0)*100:.0f}%"
                )
                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("✓ Relevant" if feedback == "like" else "Relevant", key=f"like_{job_key}", use_container_width=True):
                        st.session_state.feedback[job_key] = "like"
                        persist_feedback()
                        st.rerun()
                with b2:
                    if st.button("× Not for me" if feedback == "dislike" else "Not for me", key=f"dislike_{job_key}", use_container_width=True):
                        st.session_state.feedback[job_key] = "dislike"
                        persist_feedback()
                        st.rerun()
                with b3:
                    if st.button("★ Saved" if is_saved else "☆ Save", key=f"save_{job_key}", use_container_width=True):
                        if is_saved:
                            st.session_state.saved_jobs = [x for x in st.session_state.saved_jobs if make_job_key(x) != job_key]
                        else:
                            st.session_state.saved_jobs.append(dict(job))
                        persist_feedback()
                        st.rerun()
                if feedback == "like":
                    st.markdown('<div class="status-note">✓ You marked this opportunity as relevant.</div>', unsafe_allow_html=True)
                elif feedback == "dislike":
                    st.markdown('<div class="status-note">× You marked this opportunity as not for you.</div>', unsafe_allow_html=True)
                if is_saved:
                    st.markdown('<div class="status-note">★ Saved to your internships.</div>', unsafe_allow_html=True)
    elif not st.session_state.search_error:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.info("Your internship results will appear here. Enter your search terms and select **Search internships**.")

    if st.session_state.saved_jobs:
        with st.expander(f"Saved internships ({len(st.session_state.saved_jobs)})"):
            for saved in st.session_state.saved_jobs:
                st.markdown(f"**{saved.get('title','Opportunity')}** — {saved.get('company','Company not provided')} · {saved.get('location','')}")
                saved_description = clean_html_text(saved.get("description", ""))
                if saved_description:
                    st.caption(saved_description[:400])
                if safe_link(saved.get("url", "")):
                    st.markdown(f"[Open listing ↗]({saved['url']})")

# =============================================================================
# SAVED INTERNSHIPS
# =============================================================================
elif st.session_state.page == "Saved Internships":
    saved_jobs = st.session_state.saved_jobs
    st.markdown(f'<div class="section-title">{len(saved_jobs)} saved opportunity/opportunities</div>', unsafe_allow_html=True)
    if not saved_jobs:
        st.info("You haven't saved any internships yet. Go to Find Internships and select ☆ Save on a listing.")
    else:
        for saved_index, saved in enumerate(saved_jobs):
            saved_key = make_job_key(saved)
            with st.container(border=True):
                st.markdown(f'<div class="job-title">{html.escape(str(saved.get("title", "Opportunity")))}</div>', unsafe_allow_html=True)
                saved_meta = " · ".join(str(x) for x in [saved.get("company", ""), saved.get("location", ""), saved.get("type", ""), saved.get("salary", "")] if x)
                st.markdown(f'<div class="job-meta">{html.escape(saved_meta)}</div>', unsafe_allow_html=True)
                saved_description = clean_html_text(saved.get("description", ""))
                st.markdown(f'<div class="job-description">{html.escape(saved_description[:700])}</div>', unsafe_allow_html=True)
                s_left, s_right = st.columns([3, 1])
                with s_left:
                    if safe_link(saved.get("url", "")):
                        st.link_button("Open original listing ↗", saved["url"], use_container_width=True)
                    else:
                        st.caption("Original application link was not supplied.")
                with s_right:
                    if st.button("Remove", key=f"remove_saved_{saved_key}_{saved_index}", use_container_width=True):
                        st.session_state.saved_jobs = [
                            item for item in st.session_state.saved_jobs
                            if make_job_key(item) != saved_key
                        ]
                        persist_feedback()
                        st.rerun()

# =============================================================================
# YOUR RESUME
# =============================================================================

elif st.session_state.page == "Application Tracker":
    applications = st.session_state.applications

    st.markdown(
        f'<div class="section-title">{len(applications)} tracked applications</div>',
        unsafe_allow_html=True
    )

    if not applications:
        st.info(
            "Your tracker is empty. Open an internship from Find Internships "
            "and click Add to Application Tracker."
        )

    statuses = [
        "Interested",
        "Applied",
        "Interview",
        "Offer",
        "Rejected"
    ]

    for index, entry in enumerate(applications):
        job = entry.get("job", {})
        tracker_key = make_job_key(job)

        with st.container(border=True):
            st.markdown(
                f'<div class="job-title">{html.escape(job.get("title", "Opportunity"))}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="job-meta">{html.escape(job.get("company", "Company not provided"))} · {html.escape(job.get("location", ""))}</div>',
                unsafe_allow_html=True
            )

            current_status = entry.get("status", "Interested")
            if current_status not in statuses:
                current_status = "Interested"

            new_status = st.selectbox(
                "Application status",
                statuses,
                index=statuses.index(current_status),
                key=f"tracker_status_{tracker_key}"
            )

            new_notes = st.text_area(
                "Notes / follow-up details",
                value=entry.get("notes", ""),
                placeholder="Add interview dates, contact details or reminders...",
                key=f"tracker_notes_{tracker_key}"
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "Save changes",
                    key=f"tracker_save_{tracker_key}",
                    use_container_width=True
                ):
                    entry["status"] = new_status
                    entry["notes"] = new_notes
                    write_json(APPLICATIONS_PATH, applications)
                    st.success("Application updated!")
                    st.rerun()

            with col2:
                if st.button(
                    "Remove from tracker",
                    key=f"tracker_remove_{tracker_key}",
                    use_container_width=True
                ):
                    st.session_state.applications = [
                        item for item in applications
                        if make_job_key(item.get("job", {})) != tracker_key
                    ]
                    write_json(
                        APPLICATIONS_PATH,
                        st.session_state.applications
                    )
                    st.rerun()

            if safe_link(job.get("url", "")):
                st.link_button(
                    "Open original listing ↗",
                    job["url"],
                    use_container_width=True
                )

# =============================================================================
# AI INTERVIEW PREP
# =============================================================================
elif st.session_state.page == "AI Interview Prep":
    profile = st.session_state.profile

    # Build a unique list from internships already loaded, saved or tracked.
    available_jobs = []
    seen_job_keys = set()
    candidate_jobs = list(st.session_state.jobs) + list(st.session_state.saved_jobs)
    candidate_jobs += [
        item.get("job", {}) for item in st.session_state.applications
        if isinstance(item, dict)
    ]
    if isinstance(st.session_state.interview_job, dict):
        candidate_jobs.insert(0, st.session_state.interview_job)

    for candidate in candidate_jobs:
        if not isinstance(candidate, dict):
            continue
        candidate_key = make_job_key(candidate)
        if candidate_key not in seen_job_keys:
            available_jobs.append(candidate)
            seen_job_keys.add(candidate_key)

    if not available_jobs:
        st.info("First search for internships, save one, or add one to your Application Tracker. Then come back here to prepare for its interview.")
    else:
        labels = [
            f"{job.get('title', 'Internship')} — {job.get('company', 'Company not provided')} · {job.get('location', '')}"
            for job in available_jobs
        ]
        current_key = make_job_key(st.session_state.interview_job) if isinstance(st.session_state.interview_job, dict) else ""
        default_index = next(
            (i for i, job in enumerate(available_jobs) if make_job_key(job) == current_key),
            0
        )
        selected_index = st.selectbox(
            "Choose the internship you are preparing for",
            options=range(len(available_jobs)),
            format_func=lambda i: labels[i],
            index=default_index,
            key="interview_job_choice"
        )
        selected_interview_job = available_jobs[selected_index]
        st.session_state.interview_job = dict(selected_interview_job)

        st.markdown(
            f'<div class="surface-card"><div class="eyebrow">YOUR SELECTED INTERNSHIP</div>'
            f'<div class="job-title">{html.escape(str(selected_interview_job.get("title", "Internship")))}</div>'
            f'<div class="job-meta">{html.escape(str(selected_interview_job.get("company", "Company not provided")))} · '
            f'{html.escape(str(selected_interview_job.get("location", "")))}</div>'
            f'<div class="job-description">{html.escape(clean_html_text(selected_interview_job.get("description", ""))[:900])}</div></div>',
            unsafe_allow_html=True
        )

        question_sets = generate_interview_questions(selected_interview_job, profile)
        total_questions = sum(len(items) for items in question_sets.values())
        st.markdown(f"### Your interview practice set")
        st.caption(f"{total_questions} questions generated using this internship listing and your saved profile. These are practice prompts, not predictions of the exact interview questions.")

        for section_name, questions in question_sets.items():
            st.markdown(f"#### {section_name}")
            for number, item in enumerate(questions, start=1):
                with st.expander(f"Q{number}. {item['question']}"):
                    if item.get("why"):
                        st.write(f"**Why they may ask:** {item['why']}")
                    st.write(f"**How to approach it:** {item['tip']}")

        st.markdown("### Before the interview")
        st.write(
            "- Read the internship description again and identify its three most important requirements.\n"
            "- Prepare two project examples and be clear about your own contribution.\n"
            "- Practise explaining your answers aloud instead of memorising them.\n"
            "- If you do not know an answer, explain how you would find out."
        )
        if safe_link(selected_interview_job.get("url", "")):
            st.link_button("Revisit the original internship listing ↗", selected_interview_job["url"])

# =============================================================================
# CAREER INSIGHTS DASHBOARD
# =============================================================================
elif st.session_state.page == "Career Insights Dashboard":
    # Combine only internships the student has actually fetched, saved or tracked.
    insight_jobs = []
    seen_insight_keys = set()
    tracked_jobs = [
        item.get("job", {}) for item in st.session_state.applications
        if isinstance(item, dict) and isinstance(item.get("job"), dict)
    ]
    for job in list(st.session_state.jobs) + list(st.session_state.saved_jobs) + tracked_jobs:
        if not isinstance(job, dict):
            continue
        job_key = make_job_key(job)
        if job_key not in seen_insight_keys:
            insight_jobs.append(job)
            seen_insight_keys.add(job_key)

    applications = st.session_state.applications
    saved_count = len(st.session_state.saved_jobs)
    applied_count = sum(
        1 for item in applications
        if str(item.get("status", "")).strip().lower() in {"applied", "interview", "offer", "rejected"}
    )
    interview_count = sum(
        1 for item in applications
        if str(item.get("status", "")).strip().lower() == "interview"
    )
    offer_count = sum(
        1 for item in applications
        if str(item.get("status", "")).strip().lower() == "offer"
    )

    st.markdown("### Your activity at a glance")
    metric_cols = st.columns(4)
    metric_values = [
        ("UNIQUE OPPORTUNITIES", str(len(insight_jobs))),
        ("SAVED", str(saved_count)),
        ("APPLICATIONS SENT", str(applied_count)),
        ("INTERVIEWS", str(interview_count)),
    ]
    for col, (label, value) in zip(metric_cols, metric_values):
        with col:
            st.markdown(
                f'<div class="stat-card"><div class="stat-label">{label}</div>'
                f'<div class="stat-value">{value}</div></div>',
                unsafe_allow_html=True
            )

    if not insight_jobs:
        st.info("Your dashboard will fill with insights after you search for internships, save opportunities or add them to your Application Tracker.")
    else:
        st.markdown("### What your internship search looks like")
        role_counts = {}
        location_counts = {}
        skill_counts = {}
        match_scores = []

        def classify_role(title_text):
            value = str(title_text).lower()
            groups = {
                "AI / Machine Learning": ("machine learning", "artificial intelligence", " ai ", "deep learning", "ml intern"),
                "Data / Analytics": ("data", "analytics", "business analyst", "research analyst"),
                "Software Development": ("software", "developer", "engineering", "backend", "frontend", "full stack", "web"),
                "Web / UI Design": ("design", "ui", "ux", "graphic"),
                "Marketing / Business": ("marketing", "sales", "business development", " hr ", "human resources"),
            }
            padded = f" {value} "
            for group, terms in groups.items():
                if any(term in padded for term in terms):
                    return group
            return "Other roles"

        skill_terms = [
            ("Python", ("python",)), ("SQL", ("sql",)), ("Machine Learning", ("machine learning", " ml ")),
            ("Deep Learning", ("deep learning",)), ("NLP", ("natural language processing", " nlp ")),
            ("TensorFlow", ("tensorflow",)), ("PyTorch", ("pytorch",)), ("Pandas", ("pandas",)),
            ("NumPy", ("numpy",)), ("Data Analysis", ("data analysis", "data analytics")),
            ("Java", ("java",)), ("JavaScript", ("javascript",)), ("React", ("react",)),
            ("HTML / CSS", ("html", "css")), ("Flask", ("flask",)), ("AWS / Cloud", ("aws", "cloud")),
            ("Power BI", ("power bi",)), ("Excel", ("excel",)), ("Git", ("git",)),
            ("Computer Vision", ("computer vision", "opencv")), ("Docker", ("docker",)),
            ("Communication", ("communication",)),
        ]

        for job in insight_jobs:
            title = str(job.get("title", "Internship"))
            role = classify_role(title)
            role_counts[role] = role_counts.get(role, 0) + 1
            location = str(job.get("location", "")).strip() or "Location not specified"
            location_counts[location] = location_counts.get(location, 0) + 1
            listing_text = " ".join(str(job.get(field, "")) for field in ("title", "description", "type")).lower()
            for skill_name, aliases in skill_terms:
                if any(alias in listing_text for alias in aliases):
                    skill_counts[skill_name] = skill_counts.get(skill_name, 0) + 1
            score = job.get("match_score")
            try:
                if score is not None:
                    numeric_score = float(score)
                    if 0 <= numeric_score <= 1:
                        match_scores.append(numeric_score * 100)
                    elif 0 <= numeric_score <= 100:
                        match_scores.append(numeric_score)
            except (TypeError, ValueError):
                pass

        left, right = st.columns(2)
        with left:
            st.markdown("#### Opportunities by role")
            if role_counts:
                st.bar_chart(dict(sorted(role_counts.items(), key=lambda pair: pair[1], reverse=True)))
            else:
                st.caption("Role categories are not available yet.")
        with right:
            st.markdown("#### Locations in your shortlist")
            if location_counts:
                st.bar_chart(dict(sorted(location_counts.items(), key=lambda pair: pair[1], reverse=True)[:8]))
            else:
                st.caption("Location details are not available yet.")

        st.markdown("### Skills appearing in these listings")
        if skill_counts:
            for skill_name, count in sorted(skill_counts.items(), key=lambda pair: pair[1], reverse=True)[:10]:
                st.markdown(f"**{html.escape(skill_name)}** · mentioned in {count} of {len(insight_jobs)} opportunities")
                st.progress(count / len(insight_jobs))
        else:
            st.caption("No skills from the dashboard's current keyword list were found in the available listing descriptions.")

        st.markdown("### Your match-score snapshot")
        if match_scores:
            avg_score = sum(match_scores) / len(match_scores)
            best_score = max(match_scores)
            score_cols = st.columns(3)
            for col, (label, value) in zip(score_cols, [
                ("AVERAGE MATCH", f"{avg_score:.0f}%"),
                ("HIGHEST MATCH", f"{best_score:.0f}%"),
                ("SCORES AVAILABLE", str(len(match_scores))),
            ]):
                with col:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-label">{label}</div>'
                        f'<div class="stat-value">{value}</div></div>',
                        unsafe_allow_html=True
                    )
        else:
            st.caption("Match-score statistics will appear for opportunities that have been scored by the matching engine.")

    st.markdown("### Application outcomes")
    outcome_cols = st.columns(3)
    for col, (label, value) in zip(outcome_cols, [
        ("INTERVIEWS", str(interview_count)),
        ("OFFERS", str(offer_count)),
        ("TRACKED TOTAL", str(len(applications))),
    ]):
        with col:
            st.markdown(
                f'<div class="stat-card"><div class="stat-label">{label}</div>'
                f'<div class="stat-value">{value}</div></div>',
                unsafe_allow_html=True
            )

    st.caption("Insights are calculated only from internships currently fetched, saved or tracked in this app. They are a snapshot of your own dataset, not a representation of the entire internship market. Skill counts use keyword matching and may miss synonyms or requirements not included in short descriptions.")

# =============================================================================
# INTERNSHIP ALERTS
# =============================================================================
elif st.session_state.page == "Internship Alerts":
    alert_data = st.session_state.internship_alerts
    if not isinstance(alert_data, dict):
        alert_data = {"keywords": "", "location": "", "known_job_keys": [], "new_jobs": [], "last_checked": ""}
        st.session_state.internship_alerts = alert_data

    st.markdown("### Set up your internship alert")
    st.caption("Choose the roles and location you want to monitor. Your preferences and alert history are saved locally on this computer.")

    with st.form("internship_alert_setup", border=True):
        alert_keywords = st.text_input(
            "What internships should we look for?",
            value=str(alert_data.get("keywords") or st.session_state.profile.get("interests", "Artificial Intelligence, Data Science")),
            placeholder="e.g. NLP Intern, Machine Learning Intern, Data Analyst"
        )
        alert_location = st.text_input(
            "Preferred location",
            value=str(alert_data.get("location") or st.session_state.profile.get("preferred_location", "Gurugram")),
            placeholder="e.g. Gurugram, Delhi, Remote"
        )
        save_alert = st.form_submit_button("Save alert preferences", use_container_width=True)

    if save_alert:
        if not alert_keywords.strip():
            st.error("Please enter at least one role or keyword.")
        elif not alert_location.strip():
            st.error("Please enter a preferred location, or enter Remote if you only want remote opportunities.")
        else:
            alert_data["keywords"] = alert_keywords.strip()
            alert_data["location"] = alert_location.strip()
            alert_data["known_job_keys"] = []
            alert_data["new_jobs"] = []
            alert_data["last_checked"] = ""
            st.session_state.internship_alerts = alert_data
            write_json(ALERTS_PATH, alert_data)
            st.success("Alert preferences saved. Your next check will look for opportunities matching these preferences.")
            st.rerun()

    keywords_saved = str(alert_data.get("keywords", "")).strip()
    location_saved = str(alert_data.get("location", "")).strip()
    new_jobs = alert_data.get("new_jobs", [])
    if not isinstance(new_jobs, list):
        new_jobs = []
        alert_data["new_jobs"] = new_jobs

    st.markdown("### Your alert status")
    status_cols = st.columns(3)
    for col, (label, value) in zip(status_cols, [
        ("NEW OPPORTUNITIES", str(len(new_jobs))),
        ("SEARCH KEYWORDS", keywords_saved or "Not set"),
        ("LOCATION", location_saved or "Not set"),
    ]):
        with col:
            st.markdown(
                f'<div class="stat-card"><div class="stat-label">{label}</div>'
                f'<div class="stat-value" style="font-size:1.05rem">{html.escape(value)}</div></div>',
                unsafe_allow_html=True
            )

    if alert_data.get("last_checked"):
        st.caption(f"Last checked: {alert_data['last_checked']}")

    if not keywords_saved or not location_saved:
        st.info("Save your alert preferences above to start checking for new internships.")
    else:
        if st.button("Check for new internships now", type="primary", use_container_width=True):
            try:
                with st.spinner("Checking Jooble for matching internships..."):
                    found_jobs, found_total = jooble_search(
                        keywords_saved, location_saved, get_api_key(), 30
                    )
                    known_keys = set(alert_data.get("known_job_keys", []))
                    newly_found = []
                    for job in found_jobs:
                        if not isinstance(job, dict):
                            continue
                        key = make_job_key(job)
                        if key not in known_keys:
                            newly_found.append(job)
                            known_keys.add(key)

                    alert_data["known_job_keys"] = list(known_keys)
                    # Keep the newest 100 unreviewed alerts to avoid unbounded local files.
                    alert_data["new_jobs"] = (newly_found + new_jobs)[:100]
                    alert_data["last_checked"] = __import__("datetime").datetime.now().astimezone().strftime("%d %b %Y, %I:%M %p %Z")
                    st.session_state.internship_alerts = alert_data
                    write_json(ALERTS_PATH, alert_data)
                    st.session_state.alert_check_message = (
                        f"Found {len(newly_found)} new opportunity/opportunities among {found_total} Jooble result(s)."
                    )
            except Exception as exc:
                st.session_state.alert_check_message = ""
                st.error(f"Could not check internship alerts: {exc}")
            st.rerun()

    if st.session_state.get("alert_check_message"):
        st.success(st.session_state.alert_check_message)
        st.session_state.alert_check_message = ""

    new_jobs = alert_data.get("new_jobs", [])
    st.markdown("### New opportunities")
    if not new_jobs:
        st.info("No new opportunities waiting for you. Save your preferences and check again later.")
    else:
        if st.button("Mark all as reviewed", use_container_width=True):
            alert_data["new_jobs"] = []
            st.session_state.internship_alerts = alert_data
            write_json(ALERTS_PATH, alert_data)
            st.rerun()

        for alert_index, job in enumerate(new_jobs):
            if not isinstance(job, dict):
                continue
            with st.container(border=True):
                st.markdown(f"**{html.escape(str(job.get('title', 'Internship opportunity')))}**")
                st.caption(" · ".join(str(job.get(field, "")).strip() for field in ("company", "location", "type") if str(job.get(field, "")).strip()))
                description = clean_html_text(job.get("description", ""))
                if description:
                    st.write(description[:420] + ("…" if len(description) > 420 else ""))
                job_url = job.get("url") or job.get("link") or ""
                if safe_link(job_url):
                    st.link_button("View internship ↗", job_url, key=f"alert_open_{alert_index}")

    st.caption("Alerts are checked when you press the button; this prototype does not run in the background or send push/email notifications. Jooble results depend on the API and its current listings. Your alert preferences and history are stored in .internmatch_alerts.json.")

# SKILL GAP ANALYSIS
# =============================================================================
elif st.session_state.page == "Skill Gap Analysis":
    profile = st.session_state.profile

    available_jobs = []
    seen_job_keys = set()
    candidate_jobs = list(st.session_state.jobs) + list(st.session_state.saved_jobs)
    candidate_jobs += [
        item.get("job", {}) for item in st.session_state.applications
        if isinstance(item, dict)
    ]
    if isinstance(st.session_state.skillgap_job, dict):
        candidate_jobs.insert(0, st.session_state.skillgap_job)

    for candidate in candidate_jobs:
        if not isinstance(candidate, dict):
            continue
        candidate_key = make_job_key(candidate)
        if candidate_key not in seen_job_keys:
            available_jobs.append(candidate)
            seen_job_keys.add(candidate_key)

    if not available_jobs:
        st.info("First search for internships, save one, or add one to your Application Tracker. Then return here to compare its requirements with your profile.")
    else:
        labels = [
            f"{job.get('title', 'Internship')} — {job.get('company', 'Company not provided')} · {job.get('location', '')}"
            for job in available_jobs
        ]
        current_key = make_job_key(st.session_state.skillgap_job) if isinstance(st.session_state.skillgap_job, dict) else ""
        default_index = next(
            (i for i, job in enumerate(available_jobs) if make_job_key(job) == current_key),
            0
        )
        selected_index = st.selectbox(
            "Choose an internship to analyse",
            options=range(len(available_jobs)),
            format_func=lambda i: labels[i],
            index=default_index,
            key="skillgap_job_choice"
        )
        selected_skill_job = available_jobs[selected_index]
        st.session_state.skillgap_job = dict(selected_skill_job)

        st.markdown(
            f'<div class="surface-card"><div class="eyebrow">TARGET ROLE</div>'
            f'<div class="job-title">{html.escape(str(selected_skill_job.get("title", "Internship")))}</div>'
            f'<div class="job-meta">{html.escape(str(selected_skill_job.get("company", "Company not provided")))} · '
            f'{html.escape(str(selected_skill_job.get("location", "")))}</div></div>',
            unsafe_allow_html=True
        )

        analysis = analyse_skill_gap(selected_skill_job, profile)
        total = len(analysis["all"])
        matched_count = len(analysis["matched"])
        gap_count = len(analysis["gaps"])

        metric_cols = st.columns(3)
        metrics = [
            ("SKILLS IDENTIFIED", str(total)),
            ("IN YOUR PROFILE", str(matched_count)),
            ("TO WORK ON", str(gap_count)),
        ]
        for col, (label, value) in zip(metric_cols, metrics):
            with col:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-label">{label}</div>'
                    f'<div class="stat-value">{value}</div></div>',
                    unsafe_allow_html=True
                )

        st.markdown("### Your skill comparison")
        if not analysis["all"]:
            st.warning(
                "I couldn't identify specific skills from this listing's available description. "
                "The listing may be too short or may use requirements that are not in the current skill library."
            )
        else:
            if analysis["matched"]:
                st.markdown("#### Skills already mentioned in your profile")
                for item in analysis["matched"]:
                    st.markdown(f"- <span style='color:#087443;font-weight:700'>✓ {html.escape(item['skill'])}</span>", unsafe_allow_html=True)

            if analysis["gaps"]:
                st.markdown("#### Skills to work on")
                st.caption("These skills appeared in the internship listing but were not found in your saved skills, projects or experience.")
                for item in analysis["gaps"]:
                    with st.expander(f"＋ {item['skill']} — learning plan"):
                        st.write(f"**What to learn:** {item['learn']}")
                        st.write(f"**Mini-project to practise:** {item['project']}")
            elif analysis["all"]:
                st.success("Your profile already mentions all the skills identified in this listing. Keep practising them and prepare examples from your projects.")

        st.markdown("### Your learning roadmap")
        if analysis["gaps"]:
            st.write("Work through the missing skills one at a time. Start with the first item, build something small, and add the evidence to your profile only after you have actually practised it.")
            for step, item in enumerate(analysis["gaps"], start=1):
                st.markdown(f"**Step {step}: {item['skill']}**")
                st.write(item["learn"])
                st.caption(f"Practice task: {item['project']}")
        else:
            st.write("No specific skill gaps were detected from the available listing text. Review the full job description manually too, because this tool only recognises a curated set of common skills.")

        st.caption("This is a rule-based prototype: it identifies skills from the listing text and compares them with your saved profile. It may miss synonyms, implied requirements or skills omitted from Jooble's short description.")
        if safe_link(selected_skill_job.get("url", "")):
            st.link_button("Review the full internship requirements ↗", selected_skill_job["url"])

# =============================================================================
# YOUR RESUME
# =============================================================================
else:

    profile = st.session_state.profile
    selected_job = st.session_state.resume_job
    if not profile.get("full_name") or not profile.get("skills"):
        st.warning("Complete and save your student profile first. Your saved details will then appear here automatically.")
        if st.button("Build my profile", use_container_width=True):
            st.session_state.page = "Build Profile"
            st.rerun()
    else:
        if selected_job:
            st.markdown('<div class="eyebrow">ROLE-TARGETED RESUME</div>', unsafe_allow_html=True)
            st.markdown(f"### {html.escape(selected_job.get('title',''))}")
            st.caption(f"Tailored for {selected_job.get('company','Company not provided')} · {selected_job.get('location','')}")
        else:
            st.markdown('<div class="eyebrow">GENERAL RESUME</div>', unsafe_allow_html=True)
            st.caption("Select a role from Find Internships to tailor the resume to that opportunity.")

        resume_text = make_resume_text(profile, selected_job)
        st.markdown('<div class="section-title">Resume preview</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="resume-preview">{html.escape(resume_text)}</div>',
            unsafe_allow_html=True,
        )
        docx_data = create_docx_bytes(resume_text)
        c1, c2 = st.columns(2)
        with c1:
            if docx_data:
                st.download_button(
                    "Download resume (.docx)",
                    data=docx_data,
                    file_name=f"{re.sub(r'[^A-Za-z0-9]+','_',profile.get('full_name','Resume')).strip('_')}_Resume.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
            else:
                st.warning("Word export is unavailable because python-docx is not installed. You can download the text version.")
        with c2:
            st.download_button(
                "Download resume (.txt)",
                data=resume_text.encode("utf-8"),
                file_name="InternMatch_Resume.txt",
                mime="text/plain",
                use_container_width=True,
            )
        st.caption("Review every detail before applying. The resume uses only information saved in your profile; add measurable achievements and project details before submitting.")
        if selected_job and st.button("Clear selected role", use_container_width=True):
            st.session_state.resume_job = None
            st.rerun()

# ---------- Footer ----------
st.markdown("<div style='height:22px'></div><hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='color:#829087;font-size:.76rem;text-align:center'>INTERNMATCH AI · AI-BASED INTERNSHIP RECOMMENDATION SYSTEM · Prototype match scores are indicative, not model accuracy.</div>",
    unsafe_allow_html=True,
)

