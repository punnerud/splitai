from django.conf import settings


class NoCacheInDebugMiddleware:
    """In DEBUG, tell browsers to revalidate every response (Cache-Control:
    no-cache). Prevents stale/mixed JS + model caches on phones during the demo;
    with ETags the dev server still answers 304, so it stays cheap.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if settings.DEBUG:
            response["Cache-Control"] = "no-cache"
        return response
