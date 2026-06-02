# Gmail setup for daily PDF reports

The bot sends the end-of-day report via SMTP. Gmail does **not** accept your normal account password for apps; you need a **Google App Password**.

## 1. Turn on 2-Step Verification

1. Open [Google Account → Security](https://myaccount.google.com/security)
2. Under **How you sign in to Google**, enable **2-Step Verification** if it is off.

## 2. Create an App Password

1. Open [App passwords](https://myaccount.google.com/apppasswords) (same Google account).
2. App name: e.g. `daytrader-bot`
3. Click **Create** and copy the 16-character password (spaces optional).

## 3. Fill in `.env`

In the project root `.env`:

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your.email@gmail.com
SMTP_PASSWORD=xxxx xxxx xxxx xxxx
EMAIL_FROM=your.email@gmail.com
EMAIL_TO=your.email@gmail.com
```

- `SMTP_USERNAME` — full Gmail address used to sign in.
- `SMTP_PASSWORD` — the **App Password** from step 2 (not your regular Gmail password).
- `EMAIL_FROM` — usually the same address as `SMTP_USERNAME`.
- `EMAIL_TO` — where you want the daily report (can be the same inbox or another address).

## 4. Test

From the project folder (with dependencies installed):

```powershell
python main.py --once research
python main.py --once report
```

Check logs for `Report emailed to ...` or `Email not configured`.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Email not configured` | One of `SMTP_*` or `EMAIL_*` is empty in `.env` |
| `535 Authentication failed` | Use App Password; 2FA must be on |
| `SMTP connect failed` | Firewall; try port 587 with STARTTLS (default) |
| Report PDF missing charts | Alpaca keys must be valid; research/watchlist helps |
