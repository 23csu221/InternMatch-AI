from io import BytesIO
from docx import Document
from docx.shared import Inches, Pt
import re

def _relevance(text, target):
    terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]*", (target or "").lower()))
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]*", (text or ""))
    return sum(1 for w in words if w.lower() in terms)

def build_resume_docx(profile, job=None):
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(9.5)
    name = doc.add_heading(profile.get("full_name", "Student Name"), 0)
    name.paragraph_format.space_after = Pt(2)
    contact = " | ".join(x for x in [profile.get("email",""), profile.get("city","")] if x)
    if contact: doc.add_paragraph(contact)
    headline = profile.get("degree", "")
    if profile.get("college"): headline += " | " + profile["college"]
    if profile.get("graduation"): headline += " | Expected graduation: " + profile["graduation"]
    doc.add_paragraph(headline)
    target_text = ""
    if job:
        target_text = job.get("title","") + " " + job.get("description","")
        doc.add_heading("Target role", level=2)
        doc.add_paragraph(f"{job.get('title','')} — {job.get('company','')}")
    summary = profile.get("summary","").strip() or f"Student pursuing {profile.get('degree','')} with skills in {profile.get('skills','')}. Interested in {profile.get('domains','')}."
    doc.add_heading("Professional Summary", level=2)
    doc.add_paragraph(summary)
    skills = [x.strip() for x in profile.get("skills","").split(",") if x.strip()]
    if job: skills.sort(key=lambda x: _relevance(x, target_text), reverse=True)
    doc.add_heading("Skills", level=2)
    doc.add_paragraph(" • ".join(skills))
    if profile.get("projects","").strip():
        doc.add_heading("Projects", level=2)
        projects = [x.strip() for x in profile["projects"].split("\n\n") if x.strip()]
        if job: projects.sort(key=lambda x: _relevance(x, target_text), reverse=True)
        for project in projects: doc.add_paragraph(project, style="List Bullet")
    doc.add_heading("Education", level=2)
    doc.add_paragraph(f"{profile.get('degree','')}\n{profile.get('college','')}\nExpected graduation: {profile.get('graduation','')}")
    for title,key in [("Certifications & Courses","certifications"),("Achievements & Activities","achievements")]:
        if profile.get(key,"").strip():
            doc.add_heading(title, level=2)
            for item in profile[key].splitlines():
                if item.strip(): doc.add_paragraph(item.strip(), style="List Bullet")
    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
