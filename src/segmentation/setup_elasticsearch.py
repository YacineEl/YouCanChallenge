import os
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta
import random
from faker import Faker
import logging
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ES_HOST = os.getenv('ES_HOST', 'localhost')
ES_PORT = int(os.getenv('ES_PORT', '9200'))

def wait_for_elasticsearch(max_retries=5, retry_interval=5):
    """Wait for Elasticsearch to be ready."""
    for i in range(max_retries):
        try:
            es = Elasticsearch(f"http://{ES_HOST}:{ES_PORT}")
            if es.ping():
                logger.info("Elasticsearch is ready!")
                return es
        except Exception as e:
            logger.warning(f"Attempt {i+1}/{max_retries}: Elasticsearch not ready yet. Waiting {retry_interval} seconds...")
            time.sleep(retry_interval)
    
    raise Exception("Could not connect to Elasticsearch after multiple attempts")

es = wait_for_elasticsearch()

fake = Faker()

CATEGORIES = {
    'Electronics': [
        'wireless headphones', 'smartphone', 'laptop', 'tablet', 'smart watch',
        'gaming console', 'camera', 'bluetooth speaker', 'wireless earbuds'
    ],
    'Clothing': [
        'men\'s t-shirt', 'women\'s dress', 'jeans', 'sneakers', 'winter jacket',
        'summer shorts', 'formal shirt', 'casual shoes', 'accessories'
    ],
    'Home & Kitchen': [
        'coffee maker', 'blender', 'toaster', 'cookware set', 'kitchen utensils',
        'dinnerware', 'bedding set', 'home decor', 'storage solutions'
    ],
    'Books': [
        'fiction novels', 'self-help books', 'cookbooks', 'children\'s books',
        'business books', 'science books', 'history books', 'art books'
    ],
    'Sports': [
        'running shoes', 'yoga mat', 'fitness tracker', 'sports equipment',
        'gym accessories', 'sports clothing', 'outdoor gear', 'team sports'
    ]
}

def create_index():
    index_name = 'user_searches'
    
    try:
        if es.indices.exists(index=index_name):
            logger.info(f"Index {index_name} already exists")
            return
    except Exception as e:
        logger.error(f"Error checking if index exists: {str(e)}")
        raise
    
    mappings = {
        "mappings": {
            "properties": {
                "user_id": {"type": "keyword"},
                "search_query": {"type": "text", "analyzer": "standard"},
                "clicked_product_ids": {"type": "keyword"},
                "timestamp": {"type": "date"},
                "category": {"type": "keyword"}
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 1
        }
    }
    
    try:
        es.indices.create(index=index_name, body=mappings)
        logger.info(f"Created index {index_name}")
    except Exception as e:
        logger.error(f"Error creating index: {str(e)}")
        raise

def generate_sample_data(num_users=1000, searches_per_user=10):
    actions = []
    
    for _ in range(num_users):
        user_id = fake.uuid4()
        
        for _ in range(searches_per_user):
            category = random.choice(list(CATEGORIES.keys()))
            search_terms = CATEGORIES[category]
            
            num_searches = random.randint(1, 3)
            for _ in range(num_searches):
                search_query = random.choice(search_terms)
                
                num_clicks = random.randint(0, 3)
                clicked_products = [f"P{fake.uuid4().split('-')[0]}" for _ in range(num_clicks)]
                
                timestamp = fake.date_time_between(
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now()
                )
                
                doc = {
                    "user_id": user_id,
                    "search_query": search_query,
                    "clicked_product_ids": clicked_products,
                    "timestamp": timestamp.isoformat(),
                    "category": category
                }
                
                actions.append({
                    "_index": "user_searches",
                    "_source": doc
                })
    
    return actions

def bulk_index_data(actions):
    from elasticsearch.helpers import bulk
    
    try:
        success, failed = bulk(es, actions)
        logger.info(f"Indexed {success} documents, {failed} failed")
    except Exception as e:
        logger.error(f"Error during bulk indexing: {str(e)}")
        raise

def main():
    logger.info("Setting up Elasticsearch...")
    
    create_index()
    
    logger.info("Generating sample data...")
    actions = generate_sample_data()
    
    logger.info("Indexing sample data...")
    bulk_index_data(actions)
    
    logger.info("Setup completed successfully")

if __name__ == "__main__":
    main() 