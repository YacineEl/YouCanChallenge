import os
import sys
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from faker import Faker
from sqlalchemy import create_engine, text
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
NUM_USERS = 5000  
NUM_PRODUCTS = 500  
FREQUENCY_EVENT_PER_USER_PER_DAY = [0,3,5,10]
START_DATE = datetime.now() - timedelta(days=365)
END_DATE = datetime.now()

# Churn probabilities per week (probability of user becoming inactive)
# Each number represents the probability of churning in that week
CHURN_PROBABILITIES = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

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

def clear_database():
    try:
        logger.info("Clearing existing data...")
        session.execute(text("DELETE FROM events"))
        session.execute(text("DELETE FROM products"))
        session.execute(text("DELETE FROM users"))
        session.commit()
        logger.info("Database cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        session.rollback()
        raise

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

def generate_user_with_events(products):
    user = User(
        user_id=fake.uuid4(),
        signup_date=fake.date_time_between(start_date=START_DATE, end_date=END_DATE),
        country=fake.country_code()
    )
    session.add(user)
    session.flush()
    
    events = []
    event_types = ['viewed', 'add-to-cart', 'purchased']
    
    frequency_weights = [0.4, 0.3, 0.2, 0.1]  
    events_per_day = random.choices(FREQUENCY_EVENT_PER_USER_PER_DAY, weights=frequency_weights)[0]
    
    # Calculate total days between signup and end date
    total_days = (END_DATE - user.signup_date).days
    
    # Generate events week by week
    current_date = user.signup_date
    week_number = 0
    
    while current_date <= END_DATE and week_number < len(CHURN_PROBABILITIES):
        # Check if user has churned this week
        if random.random() < CHURN_PROBABILITIES[week_number]:
            break  # User has churned, stop generating events
        
        # Generate events for this week
        days_in_week = min(7, (END_DATE - current_date).days + 1)
        events_this_week = events_per_day * days_in_week
        
        for _ in range(events_this_week):
            # Random day within this week
            random_days = random.randint(0, days_in_week - 1)
            event_date = current_date + timedelta(days=random_days)
            
            event = Event(
                user_id=user.user_id,
                event_type=random.choices(event_types, weights=[0.7, 0.2, 0.1])[0],
                product_id=random.choice(products).product_id,
                timestamp=event_date + timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )
            )
            events.append(event)
        
        # Move to next week
        current_date += timedelta(days=7)
        week_number += 1
    
    return user, events

def populate_database():
    try:
        clear_database()
        
        logger.info("Generating products...")
        products = generate_products()
        session.bulk_save_objects(products)
        session.commit()
        logger.info(f"Generated {len(products)} products")
        
        logger.info("Generating users and events...")
        total_users = 0
        total_events = 0
        
        for _ in range(NUM_USERS):
            user, events = generate_user_with_events(products)
            session.bulk_save_objects(events)
            session.commit()
            
            total_users += 1
            total_events += len(events)
            
            if total_users % 100 == 0:
                logger.info(f"Processed {total_users} users with {total_events} events")
        
        logger.info(f"Database population completed. Generated {total_users} users and {total_events} events")
        
    except Exception as e:
        logger.error(f"Error populating database: {str(e)}")
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == "__main__":
    populate_database() 