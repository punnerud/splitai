"""Django settings for the SplitAI demo.

Local demo: DEBUG=True, sqlite3, no authentication. Users are "simulated" by each
browser picking a username (stored in localStorage and sent as a header).
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Local demo only — do not use this key in production.
SECRET_KEY = "django-insecure-splitai-local-demo-key-change-me"
DEBUG = True
# Local demo: allow all hosts so other machines on the LAN can reach the server.
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "django_extensions",  # provides runserver_plus (HTTPS for the webcam)
    "core",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "core.middleware.NoCacheInDebugMiddleware",
]

ROOT_URLCONF = "splitai.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": []},
    },
]

WSGI_APPLICATION = "splitai.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Europe/Oslo"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Feature vectors and head weights can be large JSON payloads.
DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024
