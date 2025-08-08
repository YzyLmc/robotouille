import os

def get_env_asset_path(asset_name: str, assert_exists: bool = True) -> str:
    """Return the absolute path to env asset."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    asset_dir_path = os.path.join(dir_path, "envs", "assets")
    path = os.path.join(asset_dir_path, asset_name)
    if assert_exists:
        assert os.path.exists(path), f"Env asset not found: {asset_name} under {asset_dir_path}."
    return path