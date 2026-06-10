# Portfolio & Market Dashboard (auto-updating, encrypted)

A private, password-protected dashboard for an Indian stocks + mutual-fund portfolio.
Rebuilt every 5 minutes during NSE market hours by GitHub Actions and deployed to Vercel.

## How it works
- `build.py` screens the Nifty 500 (momentum + risk + trend composite), screens mutual funds
  (AMFI/mfapi history), and reads the personal holdings panel.
- The holdings payload is **AES-256-GCM encrypted** (PBKDF2-SHA256) before being embedded in
  `template.html`, so the published file is unreadable without the password. Safe to host publicly.
- `.github/workflows/refresh.yml` runs the build on a 5-min cron and deploys `public/index.html`.

## Data freshness
Free EOD / ~15-min-delayed price data (yfinance). MF NAVs are end-of-day (AMFI). Not real-time ticks.

## Required GitHub repo secrets
| Secret | What it is |
|---|---|
| `DASH_PASSWORD` | password that unlocks the dashboard |
| `HOLDINGS_JSON` | `{"stocks":[[sym,sector,qty,avg],...],"mf":[[name,isin,cat,units,avg],...]}` |
| `VERCEL_TOKEN` | Vercel access token |
| `VERCEL_ORG_ID` | from `.vercel/project.json` after `vercel link` |
| `VERCEL_PROJECT_ID` | from `.vercel/project.json` after `vercel link` |

## Local test
```
DASH_PASSWORD=yourpass python build.py   # writes public/index.html
```
> Holdings live only in `HOLDINGS_JSON` (a secret) — never in this source. EOD/delayed data, not advice.
