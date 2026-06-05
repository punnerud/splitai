from django.db import models


class User(models.Model):
    """A "simulated" user. No authentication — just a name per browser."""

    name = models.CharField(max_length=80, unique=True)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.name


class SharedModel(models.Model):
    """A stored MLP head (the "final layers") that can be shared.

    `weights` is the JSON from `MlpHead.export_json()` in WASM. It is stored only
    on the server and is NEVER sent out to other users — they can only run
    inference.
    """

    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="models")
    name = models.CharField(max_length=120)
    classes = models.JSONField()  # list of class names, order == output index
    feat_dim = models.IntegerField()
    hidden = models.IntegerField()
    weights = models.JSONField()  # the secret head weights
    n_samples = models.IntegerField(default=0)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created"]

    def __str__(self) -> str:
        return f"{self.name} (owner: {self.owner.name})"
