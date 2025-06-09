import os
import sys
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_USER = os.getenv('DB_USER', 'itversity_retail_user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'itversity')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME','youcan')

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)

INDEXES = {
    # Indexes for weekly_active_users query
    'idx_events_week': """
    CREATE INDEX IF NOT EXISTS idx_events_week 
    ON events (date_trunc('week', timestamp));
    """,
    
    'idx_events_week_user': """
    CREATE INDEX IF NOT EXISTS idx_events_week_user 
    ON events (date_trunc('week', timestamp), user_id);
    """,
    
    'idx_events_user_id': """
    CREATE INDEX IF NOT EXISTS idx_events_user_id 
    ON events (user_id);
    """,
    
    # Indexes for revenue_per_category query
    'idx_events_event_type': """
    CREATE INDEX IF NOT EXISTS idx_events_event_type 
    ON events (event_type);
    """,
    
    'idx_events_product_id': """
    CREATE INDEX IF NOT EXISTS idx_events_product_id 
    ON events (product_id);
    """,
    
    'idx_products_product_id': """
    CREATE INDEX IF NOT EXISTS idx_products_product_id 
    ON products (product_id);
    """,
    
    'idx_events_type_product': """
    CREATE INDEX IF NOT EXISTS idx_events_type_product 
    ON events (event_type, product_id);
    """
}

def create_indexes():
    try:
        with engine.connect() as conn:
            for index_name, index_sql in INDEXES.items():
                logger.info(f"Creating index: {index_name}")
                conn.execute(text(index_sql))
                conn.commit()
                logger.info(f"Successfully created index: {index_name}")
    except Exception as e:
        logger.error(f"Error creating indexes: {str(e)}")
        raise

def drop_indexes():
    try:
        with engine.connect() as conn:
            for index_name in INDEXES.keys():
                logger.info(f"Dropping index: {index_name}")
                conn.execute(text(f"DROP INDEX IF EXISTS {index_name};"))
                conn.commit()
                logger.info(f"Successfully dropped index: {index_name}")
    except Exception as e:
        logger.error(f"Error dropping indexes: {str(e)}")
        raise

if __name__ == "__main__":

    drop_indexes()
    create_indexes()
    
    logger.info("Index creation completed successfully") 