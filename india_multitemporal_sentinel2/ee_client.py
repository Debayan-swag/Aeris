"""Earth Engine authentication helpers."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("india_s2")


def initialize_earth_engine(project: str | None = None) -> None:
    """
    Initialize the Earth Engine client.

    Prefer EE_PROJECT / config project. Does not hardcode credentials.
    """
    import ee

    project = (project or os.environ.get("EE_PROJECT") or "").strip() or None
    if not project:
        raise ValueError(
            "Earth Engine project ID is required. Set the EE_PROJECT environment variable "
            "or pass it as a parameter. See http://goo.gle/ee-auth for more information."
        )
    
    try:
        ee.Initialize(project=project)
        logger.info("Earth Engine initialized with project=%s", project)
    except Exception as first:  # noqa: BLE001
        logger.warning("ee.Initialize failed (%s); attempting interactive auth...", first)
        ee.Authenticate()
        ee.Initialize(project=project)
        logger.info("Earth Engine initialized after Authenticate()")
