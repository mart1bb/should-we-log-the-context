"""AJOUT (Renault, 11 sept. 2026) — fichier ABSENT du dépôt officiel EPFL-IMOS/hcde.

`run/main.py` fait `from src.model import GRAPH_MODELS`, mais ce paquet n'avait pas d'`__init__.py`
et `GRAPH_MODELS` n'est défini nulle part dans le dépôt. C'est un trou de packaging, pas de logique.

`GRAPH_MODELS` n'est utilisé qu'à un seul endroit :
    track_graph=cfg.task.track_graph and cfg.model.type in GRAPH_MODELS
Le dépôt ne contient AUCUN modèle de graphe (la factory n'expose que mlp/fnn/cde/hcde), et toutes
les configs ont `task.track_graph: false`. La liste vide est donc le contenu fidèle : elle
reproduit exactement le comportement voulu sans rien changer à la méthode.
"""
from .factory import BASELINE_MODELS, create_model  # noqa: F401

GRAPH_MODELS: list[str] = []
