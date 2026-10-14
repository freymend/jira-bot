from __future__ import annotations

from app import db


def main() -> None:
    db.init_db()
    print("Database migrations completed successfully.")


if __name__ == "__main__":
    main()
