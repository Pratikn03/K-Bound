"""Prospective CCT-20 data and source-model training utilities.

The package keeps target preparation and source training separate on purpose.
``prospective_data`` can extract only the leading ``images`` array from the
held-out target envelope.  ``train_source`` accepts only the named source
training and cis-validation annotation files.
"""

from .integrity import IntegrityError

__all__ = ["IntegrityError"]
