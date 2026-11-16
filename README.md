# InternMatch AI — MVP v2

A Streamlit MVP for student onboarding, Jooble internship search, profile-based ranking and tailored DOCX resume generation.

## Run on macOS

Open this folder in PyCharm, then in its Terminal run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

If `.venv` already exists, do not recreate it; just activate it and run the last two commands.

## One-time Jooble API key setup (do not share your key)

1. In PyCharm, expand the `.streamlit` folder.
2. Create a new file named `secrets.toml` (not `secrets.toml.example`).
3. Put exactly this in the file, replacing the placeholder with your own key:

```toml
JOOBLE_API_KEY = "YOUR_REAL_JOOBLE_API_KEY"
```

4. Save the file. Restart Streamlit.
5. The sidebar should show **Jooble connected**. The key will not need to be pasted each time.

Keep `secrets.toml` private. It is excluded from Git by `.gitignore`. Never send the key in chat, screenshots, or public repositories.

## What's changed in v2

- Removed Adzuna fields and setup links.
- Removed API-key entry from the app UI.
- Added one-time local Streamlit secrets configuration.
- Moved Build profile / Find internships / Your resume into the left navigation.
- Replaced bright blue styling with a warm neutral + muted green visual theme.
- Reworked internship selection: View details now immediately reruns the app and shows the selected opportunity at the top.
- Generate tailored resume now switches to the resume section.
- Kept Jooble live search and original listing links.

## Current limitations / next milestones

- Student profile and feedback are currently held in Streamlit session state; persistent accounts/database are not implemented yet.
- The resume is a DOCX draft based only on student-entered information; review and edit it before applying.
- Eligibility extraction is not yet reliable enough to declare someone eligible/ineligible. Always verify mandatory criteria on the original listing.
- Next: upload and parse existing PDF resumes, add eligibility explanations, improve semantic ranking, persist profiles/saved jobs/feedback, and validate expired listings.
- Listings depend on Jooble API coverage and availability; this app does not scrape other platforms.
