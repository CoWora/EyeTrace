"""
Make `Model` importable as a Python package.

Some persisted artifacts (e.g. `pca_model.joblib`) may contain pickled objects whose
fully-qualified class path starts with `Model.ET_model...`. Having this file ensures
those artifacts can be loaded reliably.
"""

