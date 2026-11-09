"""AJOUT (Renault, 11 sept. 2026) — fichier ABSENT du dépôt officiel EPFL-IMOS/hcde.

`run/main.py` fait `from src.train import create_trainer`, mais ni le fichier ni la fonction
n'existent dans la release. Reconstruction mécanique, sans invention de logique :

  - `src/train/train.py` ne contient que DEUX trainers : `BaseTrainer` (abstrait —
    `compute_loss` lève NotImplementedError) et `MappingTrainer` (concret). Le seul utilisable
    est donc `MappingTrainer`, et toutes les configs N-CMAPSS déclarent `task.train_type: mapping`.
  - `MappingTrainer.__init__(**kwargs)` relaie tel quel à `BaseTrainer.__init__`, dont la
    signature est (model, task, optimizer, scheduler, n_epochs, scaler, criterion, model_name,
    task_train_type, clip_grad, max_grad_norm).
  - `main.py` passe déjà model/optimizer/scheduler/criterion/scaler/task ; il ne reste qu'à
    compléter depuis la config les arguments restants, avec les valeurs que la config porte déjà.

Aucun comportement d'entraînement n'est modifié : on se contente d'instancier le trainer existant
avec les hyperparamètres déclarés dans le YAML.
"""
from ..config import cfg
from .train import BaseTrainer, MappingTrainer  # noqa: F401

TRAINERS = {
    'mapping': MappingTrainer,
}


def create_trainer(model, **kwargs):
    train_type = cfg.task.train_type
    if train_type not in TRAINERS:
        raise ValueError(
            f"train_type={train_type!r} sans trainer concret dans le dépôt. "
            f"Disponibles : {sorted(TRAINERS)}"
        )
    return TRAINERS[train_type](
        model=model,
        n_epochs=cfg.optim.max_epochs,
        task_train_type=train_type,
        clip_grad=cfg.train.clip_grad,
        max_grad_norm=cfg.train.max_grad_norm,
        **kwargs,
    )
