from django.db import models


class User(models.Model):
    """En "simulert" bruker. Ingen autentisering — kun et navn per nettleser."""

    name = models.CharField(max_length=80, unique=True)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.name


class SharedModel(models.Model):
    """Et lagret MLP-hode (de "siste lagene") som kan deles.

    `weights` er JSON-en fra `MlpHead.export_json()` i WASM. Den lagres bare pa
    serveren og sendes ALDRI ut til andre brukere — de far kun kjore inferens.
    """

    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="models")
    name = models.CharField(max_length=120)
    classes = models.JSONField()  # liste med klassenavn, rekkefolge == output-indeks
    feat_dim = models.IntegerField()
    hidden = models.IntegerField()
    weights = models.JSONField()  # de hemmelige hode-vektene
    n_samples = models.IntegerField(default=0)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created"]

    def __str__(self) -> str:
        return f"{self.name} (eier: {self.owner.name})"
