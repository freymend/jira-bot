from __future__ import annotations

import re


ASSIGNABLE_OWNER_NAMES = {"zachary": "Zachary", "zac": "Zac"}


def validated_owner_speaker(
    owner_speaker: str | None, evidence_quote: str, source_text: str | None = None
) -> str | None:
    """Allow only an eligible name explicitly present in its assignment quote."""
    owner = ASSIGNABLE_OWNER_NAMES.get((owner_speaker or "").strip().casefold())
    if owner is None:
        return None

    if not re.search(rf"\b{re.escape(owner)}\b", evidence_quote, flags=re.IGNORECASE):
        return None
    if source_text is not None:
        normalized_quote = " ".join(evidence_quote.split()).casefold()
        normalized_source = " ".join(source_text.split()).casefold()
        if normalized_quote not in normalized_source:
            return None
    return owner
