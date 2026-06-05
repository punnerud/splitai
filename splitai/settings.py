"""Django-innstillinger for SplitAI-demoen.

Lokal demo: DEBUG=True, sqlite3, ingen autentisering. Brukere "simuleres" ved at
hver nettleser velger et brukernavn (lagres i localStorage og sendes som header).
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Kun for lokal demo — ikke bruk denne nokkelen i produksjon.
SECRET_KEY = "django-insecure-splitai-local-demo-key-change-me"
DEBUG = True
# Lokal demo: tillat alle verter slik at andre maskiner paa LAN-et naar serveren.
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "django_extensions",  # gir runserver_plus (HTTPS for webkamera)
    "core",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
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

LANGUAGE_CODE = "nb-no"
TIME_ZONE = "Europe/Oslo"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Feature-vektorer og hode-vekter kan bli store JSON-payloads.
DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024
