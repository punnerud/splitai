#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "splitai.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Is the venv activated and Django installed?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
