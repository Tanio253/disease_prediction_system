from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

#SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db" # For local SQLite testing
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://dps_user:dps_password@localhost:5432/dps_db")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()