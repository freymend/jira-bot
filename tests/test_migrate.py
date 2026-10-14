from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from app import migrate


class MigrationCommandTests(unittest.TestCase):
    def test_main_confirms_successful_migrations(self) -> None:
        output = StringIO()
        with (
            patch("app.migrate.db.init_db") as init_db,
            redirect_stdout(output),
        ):
            migrate.main()

        init_db.assert_called_once_with()
        self.assertEqual(
            output.getvalue(), "Database migrations completed successfully.\n"
        )


if __name__ == "__main__":
    unittest.main()
