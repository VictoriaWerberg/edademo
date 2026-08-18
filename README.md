# Interconnect EDA

Fab / telemetry data analysis tool — ingest, validate, clean, analyse, report.

| Layer | Tech | Host |
|-------|------|------|
| Engine | Python (pandas, scipy, pandera) | local / CI |
| API | Flask + gunicorn | Heroku |
| Site | HTML + vanilla JS | Netlify |

## Milestones

| # | Description | Status |
|---|-------------|--------|
| 1 | Repo scaffold + Netlify "hello world" | ✅ done |
| 2 | Validation layer (pandera schemas) | ⬜ |
| 3 | Cleaning + join logic | ⬜ |
| 4 | Stats: ANOVA / effect size | ⬜ |
| 5 | Flask API deployed to Heroku | ⬜ |
| 6 | Site: upload form calls live API | ⬜ |
| 7 | DOE / modelling layer (stretch) | ⬜ |

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r api/requirements.txt
```

Run the API locally:

```bash
python api/app.py
```

## Deploying

**Netlify** — connect the repo, set *Base directory* to `site/`, publish directory to `.` (relative).

**Heroku** — set *Root directory* to `api/`; the `Procfile` handles startup.

Once the Heroku URL is known, update `API_URL` in `site/index.html`.
