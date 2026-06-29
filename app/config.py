from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    YOLO_MODEL_PATH: str
    KNN_MODEL_PATH: str
    DATABASE_URL: str
    VIDEO_UPLOAD_DIR : str
    PICTURE_UPLOAD_DIR : str
    VIDEO_OUTPUT_DIR : str
    PICTURE_OUTPUT_DIR : str
    
    class Config(SettingsConfigDict):
        env_file = ".env"

def get_settings():
    return Settings()