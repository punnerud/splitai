from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/users", views.users, name="users"),
    path("api/models", views.models_view, name="models"),
    path("api/infer", views.infer, name="infer"),
    path("api/yolo_tail", views.yolo_tail, name="yolo_tail"),
    path("api/yolo_head", views.yolo_head, name="yolo_head"),
]
