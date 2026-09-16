"""Validate the fixed-tokenizer contract for downstream BrainOmni.

Input is the requested tokenizer-freezing flag. Unsupported values emit a
warning and raise before model loading or artifact creation; valid requests
return without modifying configuration.
"""

import logging


logger = logging.getLogger("baseline")
FROZEN_TOKENIZER_MESSAGE = (
    "BrainOmni downstream fine-tuning requires freeze_tokenizer=True; "
    "freeze_tokenizer=False is invalid because tokenizer training is "
    "unsupported. Set model.freeze_tokenizer to true."
)


def require_frozen_tokenizer(freeze_tokenizer: bool) -> None:
    """Reject requests to train the downstream tokenizer.

    Parameters
    ----------
    freeze_tokenizer : bool
        Must be True for downstream fine-tuning or feature extraction.

    Raises
    ------
    ValueError
        If tokenizer freezing is not explicitly enabled.
    """
    if freeze_tokenizer is not True:
        logger.warning(FROZEN_TOKENIZER_MESSAGE)
        raise ValueError(FROZEN_TOKENIZER_MESSAGE)
