import os
import sys
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from faker import Faker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging

# Add parent directory to path to import database models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.setup import Base, User, Product, Event, engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Faker
fake = Faker()

# Create session
Session = sessionmaker(bind=engine)
session = Session()

# Constants
NUM_USERS = 10000
NUM_PRODUCTS = 5000
NUM_EVENTS = 1000000
START_DATE = datetime.now() - timedelta(days=365)
END_DATE = datetime.now()

# Product categories and their price ranges
CATEGORIES = {
    'Electronics': (50, 2000),
    'Clothing': (20, 200),
    'Home & Kitchen': (30, 500),
    'Books': (10, 100),
    'Sports': (25, 300),
    'Beauty': (15, 150),
    'Toys': (10, 100),
    'Food': (5, 50)
}

def generate_users():
    """Generate user data."""
    users = []
    for _ in range(NUM_USERS):
        user = User(
            user_id=fake.uuid4(),
            signup_date=fake.date_time_between(start_date=START_DATE, end_date=END_DATE),
            country=fake.country_code()
        )
        users.append(user)
    return users

def generate_products():
    """Generate product data."""
    products = []
    for _ in range(NUM_PRODUCTS):
        category = random.choice(list(CATEGORIES.keys()))
        min_price, max_price = CATEGORIES[category]
        product = Product(
            product_id=f"P{fake.uuid4().split('-')[0]}",
            category=category,
            price=round(random.uniform(min_price, max_price), 2)
        )
        products.append(product)
    return products

def generate_events(users, products):
    """Generate event data."""
    events = []
    event_types = ['viewed', 'add-to-cart', 'purchased']
    
    for _ in range(NUM_EVENTS):
        user = random.choice(users)
        product = random.choice(products)
        event = Event(
            user_id=user.user_id,
            event_type=random.choices(event_types, weights=[0.7, 0.2, 0.1])[0],
            product_id=product.product_id,
            timestamp=fake.date_time_between(
                start_date=max(user.signup_date, START_DATE),
                end_date=END_DATE
            )
        )
        events.append(event)
    return events

def populate_database():
    """Populate the database with generated data."""
    try:
        logger.info("Generating users...")
        users = generate_users()
        session.bulk_save_objects(users)
        session.commit()
        logger.info(f"Generated {len(users)} users")

        logger.info("Generating products...")
        products = generate_products()
        session.bulk_save_objects(products)
        session.commit()
        logger.info(f"Generated {len(products)} products")

        logger.info("Generating events...")
        events = generate_events(users, products)
        # Process events in batches to avoid memory issues
        batch_size = 10000
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            session.bulk_save_objects(batch)
            session.commit()
            logger.info(f"Processed {i + len(batch)} events")

        logger.info("Database population completed successfully")
    except Exception as e:
        logger.error(f"Error populating database: {str(e)}")
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == "__main__":
    populate_database() 