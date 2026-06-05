"""API-endepunkter for SplitAI.

Ingen autentisering. "Innlogget" bruker bestemmes av headeren `X-User` (settes av
nettleseren fra localStorage), slik at ulike nettlesere = ulike brukere.
"""
from __future__ import annotations

import json

from django.conf import settings
from django.http import FileResponse, HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .head_runtime import run_head
from .models import SharedModel, User


def _current_user(request: HttpRequest) -> User | None:
    name = (request.headers.get("X-User") or "").strip()
    if not name:
        return None
    user, _ = User.objects.get_or_create(name=name[:80])
    return user


def _body(request: HttpRequest) -> dict:
    if not request.body:
        return {}
    return json.loads(request.body.decode("utf-8"))


def index(request: HttpRequest) -> FileResponse:
    # Samme statiske index.html som GitHub Pages bruker (relative stier).
    return FileResponse(open(settings.BASE_DIR / "index.html", "rb"))


@require_http_methods(["GET"])
def users(request: HttpRequest) -> JsonResponse:
    data = [
        {"name": u.name, "models": u.models.count()}
        for u in User.objects.all().order_by("name")
    ]
    return JsonResponse({"users": data})


@csrf_exempt
@require_http_methods(["GET", "POST"])
def models_view(request: HttpRequest) -> JsonResponse:
    if request.method == "GET":
        # Merk: vi sender ALDRI med `weights` her — bare metadata.
        data = [
            {
                "id": m.id,
                "name": m.name,
                "owner": m.owner.name,
                "classes": m.classes,
                "feat_dim": m.feat_dim,
                "hidden": m.hidden,
                "n_samples": m.n_samples,
                "created": m.created.isoformat(),
            }
            for m in SharedModel.objects.select_related("owner").all()
        ]
        return JsonResponse({"models": data})

    # POST: lagre et nytt hode for gjeldende bruker.
    user = _current_user(request)
    if user is None:
        return JsonResponse({"error": "mangler X-User"}, status=400)

    payload = _body(request)
    try:
        weights = json.loads(payload["weights"])  # JSON-streng fra export_json()
        name = (payload.get("name") or f"{user.name} sin modell").strip()[:120]
        classes = payload["classes"]
    except (KeyError, json.JSONDecodeError) as exc:
        return JsonResponse({"error": f"ugyldig payload: {exc}"}, status=400)

    if len(classes) != int(weights["classes"]):
        return JsonResponse(
            {"error": "antall klassenavn matcher ikke vektene"}, status=400
        )

    model = SharedModel.objects.create(
        owner=user,
        name=name,
        classes=classes,
        feat_dim=int(weights["feat"]),
        hidden=int(weights["hidden"]),
        weights=weights,
        n_samples=int(payload.get("n_samples", 0)),
    )
    return JsonResponse({"id": model.id, "name": model.name})


@csrf_exempt
@require_http_methods(["POST"])
def infer(request: HttpRequest) -> JsonResponse:
    """Kjor de siste lagene pa serveren ut fra features klienten sendte inn."""
    payload = _body(request)
    try:
        model = SharedModel.objects.select_related("owner").get(
            id=int(payload["model_id"])
        )
        feat = payload["features"]
    except (KeyError, ValueError, SharedModel.DoesNotExist):
        return JsonResponse({"error": "ukjent modell eller mangler features"}, status=400)

    try:
        probs = run_head(model.weights, feat)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    ranked = sorted(
        ({"label": model.classes[i], "prob": float(p)} for i, p in enumerate(probs)),
        key=lambda d: d["prob"],
        reverse=True,
    )
    return JsonResponse(
        {
            "model": model.name,
            "owner": model.owner.name,
            "predictions": ranked,
            "note": "Hode-vektene ble kjort pa serveren og ble aldri sendt til klienten.",
        }
    )
