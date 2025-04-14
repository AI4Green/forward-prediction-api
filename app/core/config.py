import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment-specific overrides.
    """

    # global
    version: str = Field(default="1", alias="VERSION")

    model_dir: str = Field(default="trained_models", alias="MODEL_DIR")
    model_filename: str = Field(default="model.pt", alias="MODEL_FILENAME")
    use_cuda_if_available: bool = Field(default=False, alias="USE_CUDA_IF_AVAILABLE")

    # auth
    auth_enabled: bool = Field(default=False, alias="AUTH_ENABLED")
    api_key: str = Field(default="", alias="API_KEY")
    api_key_header: str = Field(default="X-API-Key", alias="API_KEY_HEADER")

    # env config
    env: str = Field(default="development", alias="PYTHON_ENV")
    debug: bool = True
    round_digit: int = 3

    # derived property
    @property
    def device(self) -> str:
        if self.use_cuda_if_available and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # pydantic configuration
    model_config = SettingsConfigDict(
        env_file=".env.local", extra="ignore", case_sensitive=False
    )
