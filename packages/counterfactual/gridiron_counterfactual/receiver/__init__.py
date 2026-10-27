"""The receiver what-if: at the moment of the throw, was there a better receiver?

Connor's model, alongside the play model in this package. Where ``decision_engine``
forecasts where everyone will be while the ball is in the air (the 2026 release), this
scores every receiver at the instant of the throw on the 2025 release, which carries all
22 players and the game situation. What it adds is a value in expected points and win
probability, an availability filter, and scoring by models that never saw the outcome of
the play they judge. See ``README.md`` in this folder.

``bdb2025``   the 2025 release: per-game cache, tables, ``PlayTrack``
``targets``   every receiver at the throw, as a feature table
``value``     catch chance, yards and expected points for a receiver
``whatif``    availability, uncertainty, cross-fitting; the flag and its record
``winprob``   expected points to win probability, learned from the release
``simulate``  the alternative as a ``PlayTrack``: real players, an invented ball
``tackles``   missed tackles from stream 2's interaction layer (first version)
``export``    the records and simulated plays ``apps/api`` serves

    python -m gridiron_counterfactual.receiver --help
"""
