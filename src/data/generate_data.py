import os
import sys
import random
from datetime import datetime, timedelta
from faker import Faker
from sqlalchemy.orm import sessionmaker
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.setup import Base, User, Product, Event, engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fake = Faker()
Session = sessionmaker(bind=engine)
session = Session()

def create_user():
    user = User(
        user_id=fake.uuid4(),
        signup_date=fake.date_time_between(start_date=datetime.now() - timedelta(days=365), end_date=datetime.now()),
        country=fake.country_code()
    )
    return user

def create_product():
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Sports', 'Beauty', 'Toys', 'Food']
    category = random.choice(categories)
    
    if category == 'Electronics':
        price = random.uniform(50, 2000)
    elif category == 'Clothing':
        price = random.uniform(20, 200)
    elif category == 'Home & Kitchen':
        price = random.uniform(30, 500)
    elif category == 'Books':
        price = random.uniform(10, 100)
    elif category == 'Sports':
        price = random.uniform(25, 300)
    elif category == 'Beauty':
        price = random.uniform(15, 150)
    elif category == 'Toys':
        price = random.uniform(10, 100)
    else:  # Food
        price = random.uniform(5, 50)
    
    product = Product(
        product_id=f"P{fake.uuid4().split('-')[0]}",
        category=category,
        price=round(price, 2)
    )
    return product

def create_event(user, product):
    event_types = ['viewed', 'add-to-cart', 'purchased']
    event_type = random.choice(event_types)
    
    event = Event(
        user_id=user.user_id,
        event_type=event_type,
        product_id=product.product_id,
        timestamp=fake.date_time_between(
            start_date=max(user.signup_date, datetime.now() - timedelta(days=365)),
            end_date=datetime.now()
        )
    )
    return event

def add_data_to_database():
    try:
        # Create users
        logger.info("Creating users...")
        for _ in range(1000):
            user = create_user()
            session.add(user)
            session.commit()
        logger.info(f"{len(users)} Users created")

        # Create products
        logger.info("Creating products...")
        for _ in range(100):
            product = create_product()
            session.add(product)
            session.commit()
        logger.info(f"{len(products)} Products created")

        # Create events
        logger.info("Creating events...")
        users = session.query(User).all()
        products = session.query(Product).all()
        
        for _ in range(10000):
            user = random.choice(users)
            product = random.choice(products)
            event = create_event(user, product)
            session.add(event)
            session.commit()
        logger.info("Events created")

        logger.info("All data added successfully")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    add_data_to_database() 