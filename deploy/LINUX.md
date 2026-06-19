# Linux deployment (systemd)

One repo, same code as Windows. Only `.env` and `data/` are per-machine.

## 1. Clone and install Python deps

```bash
git clone git@github.com:dunryan/daytrader-bot.git ~/daytrader-bot
cd ~/daytrader-bot

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. Secrets and smoke test

```bash
cp .env.example .env
chmod 600 .env
nano .env   # Alpaca keys at minimum

mkdir -p data logs
source .venv/bin/activate
python main.py --once research
python main.py --once trade
```

If research returns symbols on a trading day, data + config are wired correctly.

## 3. Install systemd service

From the repo root (adjust `RUN_USER` if you use a dedicated `trader` account):

```bash
chmod +x deploy/install-linux-systemd.sh
sudo INSTALL_DIR="$HOME/daytrader-bot" RUN_USER="$USER" ./deploy/install-linux-systemd.sh
```

Or install to `/opt` with a dedicated user:

```bash
sudo useradd -r -m -s /bin/bash trader || true
sudo git clone git@github.com:dunryan/daytrader-bot.git /opt/daytrader-bot
sudo chown -R trader:trader /opt/daytrader-bot
# ... venv + .env as trader ...
sudo INSTALL_DIR=/opt/daytrader-bot RUN_USER=trader ./opt/daytrader-bot/deploy/install-linux-systemd.sh
```

## 4. Day-to-day commands

```bash
sudo systemctl status daytrader
journalctl -u daytrader -f              # live logs
sudo systemctl restart daytrader        # after git pull or config change
sudo systemctl stop daytrader           # halt (e.g. holiday)
```

## 5. After code updates

```bash
cd ~/daytrader-bot
git pull
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/check_db.py          # integrity + sklearn + meta model
sudo systemctl restart daytrader
```

If `check_db.py` reports corruption, stop the service first:

```bash
sudo systemctl stop daytrader
python scripts/check_db.py --backup --vacuum
# if still failing, dump/restore — see script output
sudo systemctl start daytrader
```

Verify meta-label shadow loads (needs scikit-learn + OOS model):

```bash
python -c "import sklearn; from pathlib import Path; print('sklearn ok')"
ls -la data/models/meta_label_core10_oos.pkl
grep model_path config/config.yaml
```

## Troubleshooting

| Symptom | Check |
|---------|--------|
| Service fails immediately | `journalctl -u daytrader -n 50` |
| No watchlist | Alpaca keys in `.env`; `python main.py --once research`; grep `SCREENER` in logs |
| Empty watchlist every day | Fixed in TOD RVOL screener — `git pull`; re-run research |
| `database disk image is malformed` | `sudo systemctl stop daytrader`; `python scripts/check_db.py` |
| Meta filter inert | `pip install scikit-learn`; confirm `meta_label_core10_oos.pkl` exists |
| Permission errors | Service `User` must own `data/`, `logs/`, `.env` |
| Wrong schedule times | `config/config.yaml` → `app.timezone: America/New_York` |

Manual unit edit (without the script):

```bash
sudo nano /etc/systemd/system/daytrader.service
sudo systemctl daemon-reload
sudo systemctl restart daytrader
```
