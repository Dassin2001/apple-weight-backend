from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from app.models import ProcessingStatus
from app.config import get_settings

SQLALCHEMY_DATABASE_URL = get_settings().DATABASE_URL

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class VideoProcessingHistory(Base):
    __tablename__ = "video_processing_history"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    upload_time = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float, nullable=True)
    unique_apple_count = Column(Integer, nullable=True)
    total_weight = Column(Float, nullable=True)
    output_path = Column(String, nullable=True)
    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PROCESSING)
    error_message = Column(String, nullable=True)

class PictureProcessingHistory(Base):
    __tablename__ = "picture_processing_history"

    id = Column(Integer, primary_key=True, index=True)
    filenames = Column(String, index=True)
    task_id = Column(String, index=True)
    upload_time = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float, nullable=True)
    unique_apple_count = Column(Integer, nullable=True)
    total_weight = Column(Float, nullable=True)
    output_path = Column(String, nullable=True)
    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PROCESSING)
    error_message = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()