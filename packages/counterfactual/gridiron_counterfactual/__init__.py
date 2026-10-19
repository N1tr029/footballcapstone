"""Stream 3's side of the seams — the play model, joined to the shared contract.

``decision_engine.py`` beside this package is Landon's prototype, merged with its
own history: it trains on the Big Data Bowl 2026 release, forecasts where everyone
will be while the ball is in the air, and writes a typed judgement per play. It
speaks the competition's dialect — its column names, its coordinates, its compass
angles, its own report shape.

Nothing else in this repository speaks that dialect. The renderer draws a
``PlayTrack``; the vision pipeline produces a ``PlayTrack``; ``docs/roles.md``
specifies a ``Recommendation``. These four modules are the translation, and they
are deliberately the *only* place the translation happens:

``bdb``             the 2026 release -> ``PlayTrack``, so the model's own training
                    data animates in the renderer like anything else.
``features``        ``PlayTrack`` -> the model's feature rows. This is the join that
                    matters: it is what lets the model score the vision pipeline's
                    output instead of only Kaggle's, and it has to undo every
                    transform ``bdb`` applied or the model is fed a mirror image of
                    the world it was trained on.
``forecast``        predicted coordinates -> a ``PlayTrack`` marked SIMULATED.
``recommendation``  a ``decision_engine`` report -> the ``Recommendation`` contract.

The honest caveats live with the code that earns them; see each module.
"""

from .recommendation import Recommendation, from_report, validate

__all__ = ["Recommendation", "from_report", "validate"]
