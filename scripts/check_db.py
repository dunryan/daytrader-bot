#!/usr/bin/env python3
"""Verify SQLite health, ML deps, and model artifacts for paper sim.

Usage:
    python scripts/check_db.py
    python scripts/check_db.py --vacuum
    python scripts/check_db.py --db data/trading.db
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from daytrader.config.settings import get_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check trading.db integrity and sim prerequisites")
    p.add_argument("--db", default=None, help="SQLite path (default: config app.db_path)")
    p.add_argument("--vacuum", action="store_true", help="run VACUUM if integrity ok")
    p.add_argument("--backup", action="store_true", help="copy db to .bak before vacuum")
    return p.parse_args()


def _resolve_db(path: str | None) -> Path:
    if path:
        return Path(path)
    settings = get_settings()
    db = Path(settings.app.db_path)
    if not db.is_absolute():
        db = Path(__file__).resolve().parents[1] / db
    return db


def check_integrity(db_path: Path) -> tuple[bool, str]:
    if not db_path.exists():
        return False, f"database not found: {db_path}"
    try:
        con = sqlite3.connect(db_path)
        row = con.execute("PRAGMA integrity_check").fetchone()
        con.close()
    except sqlite3.DatabaseError as exc:
        return False, str(exc)
    result = row[0] if row else "unknown"
    return result == "ok", result


def check_ml(root: Path) -> list[str]:
    issues: list[str] = []
    try:
        import sklearn  # noqa: F401
    except ImportError:
        issues.append("scikit-learn not installed — run: pip install -r requirements.txt")

    settings = get_settings()
    model = root / settings.ml.meta_filter.model_path
    if not model.exists():
        issues.append(f"meta model missing: {model}")
    return issues


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    db_path = _resolve_db(args.db)

    print(f"DB: {db_path}")
    ok, msg = check_integrity(db_path)
    print(f"  integrity: {'OK' if ok else 'FAIL'} ({msg})")

    for extra in (Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
        if extra.exists() and not ok:
            print(f"  note: journal file present ({extra.name}) — stop bot before repair")

    ml_issues = check_ml(root)
    if ml_issues:
        print("ML / deps:")
        for issue in ml_issues:
            print(f"  - {issue}")
    else:
        settings = get_settings()
        print(f"ML: OK ({settings.ml.meta_filter.model_path}, scikit-learn importable)")

    if not ok:
        print()
        print("Repair options (stop daytrader first):")
        print(f"  cp {db_path} {db_path}.corrupt.bak")
        print(f"  sqlite3 {db_path} \".dump\" | sqlite3 {db_path}.new && mv {db_path}.new {db_path}")
        raise SystemExit(1)

    if args.vacuum:
        if args.backup:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = db_path.with_suffix(f".{stamp}.bak")
            shutil.copy2(db_path, backup)
            print(f"  backup: {backup}")
        con = sqlite3.connect(db_path)
        con.execute("VACUUM")
        con.close()
        print("  vacuum: done")

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
