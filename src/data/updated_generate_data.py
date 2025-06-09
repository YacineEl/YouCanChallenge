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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.setup import Base, User, Product, Event, engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fake = Faker()

Session = sessionmaker(bind=engine)
session = Session()

NUM_USERS = 10000
NUM_PRODUCTS = 5000
NUM_EVENTS = 1000000
START_DATE = datetime.now() - timedelta(days=365)
END_DATE = datetime.now()

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

def generate_user_activity(user, products, start_date, end_date):
    events = []
    event_types = ['viewed', 'add-to-cart', 'purchased']
    
    activity_level = random.choices(['high', 'medium', 'low'], weights=[0.2, 0.5, 0.3])[0]
    
    if activity_level == 'high':
        sessions_per_week = random.randint(3, 7)
        events_per_session = random.randint(5, 15)
        retention_probability = 0.9  # 90% chance of returning each week
    elif activity_level == 'medium':
        sessions_per_week = random.randint(1, 3)
        events_per_session = random.randint(3, 8)
        retention_probability = 0.7 
    else:  
        sessions_per_week = random.randint(0, 2)
        events_per_session = random.randint(1, 5)
        retention_probability = 0.4  
    
    current_date = start_date
    while current_date <= end_date:
        # Check if user is active this week
        if random.random() < retention_probability:
            # Generate sessions for this week
            for _ in range(sessions_per_week):
                session_start = current_date + timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )
                
                # Generate events for this session
                for _ in range(events_per_session):
                    event = Event(
                        user_id=user.user_id,
                        event_type=random.choices(event_types, weights=[0.7, 0.2, 0.1])[0],
                        product_id=random.choice(products).product_id,
                        timestamp=session_start + timedelta(minutes=random.randint(1, 30))
                    )
                    events.append(event)
        
        current_date += timedelta(days=7)
    
    return events

def generate_events(users, products):
    all_events = []
    
    for user in users:
        user_events = generate_user_activity(
            user,
            products,
            user.signup_date,
            END_DATE
        )
        all_events.extend(user_events)
    
    return all_events

def populate_database():
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