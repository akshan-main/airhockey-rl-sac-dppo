"""Air hockey environment, physics, policies, and training pipeline.

This package is intentionally empty at the top level so individual
submodules can be imported in minimal environments (e.g. the FastAPI
backend's Docker image only installs the storage dependencies, not
torch or gymnasium). Import directly from submodules:

    from airhockey.env import AirHockeyEnv
    from airhockey.storage import HFBucketStore
    from airhockey.sac import SACAgent
    from airhockey.policy import UNet1D
"""
