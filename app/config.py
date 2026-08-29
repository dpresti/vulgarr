from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SPF_", env_file=".env", extra="ignore")

    # Storage
    data_dir: Path = Path("/data")
    media_root: Path = Path("/media")

    # Sonarr / Radarr / Bazarr integration
    sonarr_url: str = ""
    sonarr_api_key: str = ""
    radarr_url: str = ""
    radarr_api_key: str = ""
    bazarr_url: str = ""
    bazarr_api_key: str = ""
    doesthedogdie_api_key: str = ""

    # Shared secret required on inbound webhook calls (?token=...)
    webhook_token: str = ""

    # Processing defaults (overridable at runtime via the Settings page / DB)
    default_concurrency: int = 2
    default_subtitle_language: str = "en"

    # ffmpeg
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    clean_track_title: str = "Clean"
    clean_track_language: str = "eng"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def database_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.db_path}"


settings = Settings()
