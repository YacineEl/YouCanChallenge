import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# db conn
DB_USER = os.getenv('DB_USER', 'itversity_retail_user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'itversity')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'youcan')

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    user_id = Column(String, primary_key=True)
    signup_date = Column(DateTime, nullable=False)
    country = Column(String, nullable=False)

class Product(Base):
    __tablename__ = 'products'
    
    product_id = Column(String, primary_key=True)
    category = Column(String, nullable=False)
    price = Column(Float, nullable=False)

class Event(Base):
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey('users.user_id'), nullable=False)
    event_type = Column(String, nullable=False)  # viewed, add-to-cart, purchased
    product_id = Column(String, ForeignKey('products.product_id'), nullable=False)
    timestamp = Column(DateTime, nullable=False)

def create_tables():
    try:
        Base.metadata.create_all(engine)
        logger.info("Successfully created all tables")
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        raise

def init_db():
    create_tables()

if __name__ == "__main__":
    init_db() 