"""API endpoints for SplitAI.

No authentication. The "logged-in" user is determined by the `X-User` header
(set by the browser from localStorage), so different browsers = different users.
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
    # The same static index.html that GitHub Pages uses (relative paths).
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
        # Note: we NEVER include `weights` here — only metadata.
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

    # POST: save a new head for the current user.
    user = _current_user(request)
    if user is None:
        return JsonResponse({"error": "missing X-User"}, status=400)

    payload = _body(request)
    try:
        weights = json.loads(payload["weights"])  # JSON string from export_json()
        name = (payload.get("name") or f"{user.name}'s model").strip()[:120]
        classes = payload["classes"]
    except (KeyError, json.JSONDecodeError) as exc:
        return JsonResponse({"error": f"invalid payload: {exc}"}, status=400)

    if len(classes) != int(weights["classes"]):
        return JsonResponse(
            {"error": "number of class names does not match the weights"}, status=400
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
    """Run the final layers on the server from the features the client sent in."""
    payload = _body(request)
    try:
        model = SharedModel.objects.select_related("owner").get(
            id=int(payload["model_id"])
        )
        feat = payload["features"]
    except (KeyError, ValueError, SharedModel.DoesNotExist):
        return JsonResponse({"error": "unknown model or missing features"}, status=400)

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
            "note": "The head weights ran on the server and were never sent to the client.",
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def yolo_tail(request: HttpRequest) -> JsonResponse:
    """Run the last YOLO decode ops (model B) on the intermediate tensors the
    client computed with model A. Returns detections (640-space) + server_ms."""
    payload = _body(request)
    try:
        boxes = payload["boxes"]
        scores = payload["scores"]
        indices = payload["indices"]
        thresh = float(payload.get("score_thresh", 0.3))
    except KeyError:
        return JsonResponse({"error": "missing tensors"}, status=400)

    try:
        from .yolo_tail import run_tail

        dets, server_ms = run_tail(boxes, scores, indices, thresh)
    except Exception as exc:  # onnxruntime missing, bad shapes, …
        return JsonResponse({"error": str(exc)}, status=400)

    return JsonResponse({"detections": dets, "server_ms": round(server_ms, 3)})


@csrf_exempt
@require_http_methods(["POST"])
def yolo_head(request: HttpRequest) -> JsonResponse:
    """Early-cut split: the client sent int8 P3/P4/P5 (body) + {shapes,scales,thresh}
    (X-Meta header). Run the detection head (model B2) here and return detections."""
    try:
        meta = json.loads(request.headers.get("X-Meta", "{}"))
        shapes = meta["shapes"]
        scales = meta["scales"]
        thresh = float(meta.get("thresh", 0.3))
    except (KeyError, json.JSONDecodeError):
        return JsonResponse({"error": "missing/invalid X-Meta"}, status=400)

    try:
        from .yolo_head import run_head

        dets, server_ms, used = run_head(request.body, shapes, scales, thresh)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    return JsonResponse(
        {"detections": dets, "server_ms": round(server_ms, 3), "bytes": used}
    )
