#!/usr/bin/env python
"""Django sin kommandolinje for administrasjon."""
import os
import sys


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "splitai.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Fikk ikke importert Django. Er venv aktivert og Django installert?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
