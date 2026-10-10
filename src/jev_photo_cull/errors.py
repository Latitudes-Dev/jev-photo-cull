"""Error types. Config problems are usage errors; backend failures are runtime errors."""


class PhotoCullError(Exception):
    """Base error for scan failures that should stop the run."""


class ConfigError(PhotoCullError):
    """Missing credentials, bad flags, or an output path that would touch inputs."""
