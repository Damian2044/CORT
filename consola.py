from __future__ import annotations

import warnings


def filtrar_warnings_esperados() -> None:
    """Oculta advertencias esperadas de la heuristica SOTA sin cambiar la logica."""
    warnings.filterwarnings("ignore", message="Mean of empty slice.*")
    warnings.filterwarnings("ignore", message="invalid value encountered in divide.*")
    warnings.filterwarnings(
        "ignore",
        message="The number of unique classes is greater than 50% of the number of samples.*",
    )
