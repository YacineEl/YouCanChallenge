import os
import logging
from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VISUALIZATION_DIR = os.path.join('src', 'segmentation' ,'visualization')
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

load_dotenv()

ES_HOST = os.getenv('ES_HOST', 'localhost')
ES_PORT = int(os.getenv('ES_PORT', '9200'))

es = Elasticsearch(
    f"http://{ES_HOST}:{ES_PORT}",
    verify_certs=False
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_embeddings(texts):
    # Get embeddings for a list of texts using the model
    try:
        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**encoded_input)
        
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings
    except Exception as e:
        logger.error(f"Error getting embeddings: {str(e)}")
        raise

def get_user_search_data(days=30):
    # Retrieve user search data from Elasticsearch
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Query Elasticsearch
    query = {
        "size": 10000,
        "query": {
            "range": {
                "timestamp": {
                    "gte": start_date.isoformat(),
                    "lte": end_date.isoformat()
                }
            }
        }
    }
    
    response = es.search(index="user_searches", body=query)
    logger.info(f"Retrieved {len(response['hits']['hits'])} search records")
    
    hits = response['hits']['hits']
    data = []
    
    for hit in hits:
        source = hit['_source']
        data.append({
            'user_id': source['user_id'],
            'search_query': source['search_query'],
            'category': source['category'],
            'clicked_products': len(source.get('clicked_product_ids', [])),
            'timestamp': source['timestamp']
        })
    
    pd.DataFrame(data).to_csv(os.path.join(VISUALIZATION_DIR, "userSearchData.csv"))
    return pd.DataFrame(data)

def get_search_embeddings(df):
    # Convert search queries to embeddings using the model
    unique_queries = df['search_query'].unique()
    embeddings = get_embeddings(unique_queries.tolist())
    
    # Create mapping
    query_embeddings = dict(zip(unique_queries, embeddings))
    df['query_embedding'] = df['search_query'].map(query_embeddings)
    
    return df

def create_user_features(df):
    # Create user-level features from search data with embeddings.
    def average_embeddings(embeddings):
        return np.mean(np.vstack(embeddings.values), axis=0)
    
    user_features = df.groupby('user_id').agg({
        'search_query': 'count',
        'clicked_products': 'sum',
        'category': lambda x: x.value_counts().to_dict(),
        'query_embedding': average_embeddings  
    }).reset_index()
    
    user_features['search_click_ratio'] = user_features['clicked_products'] / user_features['search_query']
    
    user_features['main_category'] = user_features['category'].apply(
        lambda x: max(x.items(), key=lambda y: y[1])[0] if isinstance(x, dict) and len(x) > 0 else 'unknown'
    )
    
    logger.info(f"Created features for {len(user_features)} users")
    user_features.to_csv(os.path.join(VISUALIZATION_DIR, 'UserFeatures.csv'))
    return user_features


def reduce_embedding_dimension(user_features, n_components=5):
    # Reduce embedding dimensionality using PCA with safety checks.
    try:
        # Extract embeddings matrix
        embeddings = np.vstack(user_features['query_embedding'].values)
        logger.info(f"Embedding shape: {embeddings.shape}")
        
        # Determine safe n_components
        safe_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
        if safe_components < n_components:
            logger.warning(f"Reducing PCA components to {safe_components} due to data shape")
        
        # Apply PCA
        pca = PCA(n_components=safe_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Add PCA components to features
        for i in range(safe_components):
            user_features[f'embedding_pc_{i+1}'] = reduced_embeddings[:, i]
        
        # Log explained variance
        explained_var = sum(pca.explained_variance_ratio_)
        logger.info(f"PCA explained variance: {explained_var:.2%} with {safe_components} components")
        
        return user_features, safe_components
    except Exception as e:
        logger.error(f"Error in dimensionality reduction: {str(e)}")
        return user_features, 0

def perform_clustering(user_features, semantic_features_count, n_clusters=5):
    # Perform K-means clustering with combined features.
    try:
        behavior_features = ['search_query', 'clicked_products', 'search_click_ratio']
        semantic_features = [f'embedding_pc_{i+1}' for i in range(semantic_features_count)]
        all_features = behavior_features + semantic_features
        
        features = user_features[all_features].values
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(user_features)))
        user_features['cluster'] = kmeans.fit_predict(scaled_features)
        
        return user_features
    except Exception as e:
        logger.error(f"Clustering failed: {str(e)}")
        user_features['cluster'] = 0
        return user_features

def analyze_segments(user_features):
    """Analyze segments with semantic insights."""
    segments = []
    
    for cluster in user_features['cluster'].unique():
        segment_data = user_features[user_features['cluster'] == cluster]
        
        # Basic characteristics
        segment = {
            'cluster': cluster,
            'size': len(segment_data),
            'avg_searches': segment_data['search_query'].mean(),
            'avg_clicks': segment_data['clicked_products'].mean(),
            'avg_click_ratio': segment_data['search_click_ratio'].mean(),
            'main_categories': segment_data['main_category'].value_counts().head(3).to_dict()
        }
        
        semantic_cols = [col for col in user_features if col.startswith('embedding_pc_')]
        if semantic_cols:
            segment['avg_embedding'] = segment_data[semantic_cols].mean().values
        
        if segment['avg_click_ratio'] > 0.7:
            segment['name'] = 'High-Intent Shoppers'
        elif segment['avg_searches'] > 20:
            segment['name'] = 'Research-Oriented Users'
        elif segment['avg_click_ratio'] < 0.3:
            segment['name'] = 'Window Shoppers'
        elif semantic_cols and 'avg_embedding' in segment:
            if np.argmax(segment['avg_embedding']) == 0:
                segment['name'] = 'Tech-Focused Users'
            elif np.argmax(segment['avg_embedding']) == 1:
                segment['name'] = 'Fashion-Focused Users'
            else:
                segment['name'] = 'Semantic Group ' + str(cluster)
        else:
            segment['name'] = 'Balanced Users'
        
        segments.append(segment)
    
    return segments

def visualize_segments(user_features, segments):
    """Create comprehensive visualizations for user segments."""
    try:
        # Set up the plot style
        plt.style.use('seaborn')
        
        # 1. Segment Distribution (Pie Chart)
        plt.figure(figsize=(10, 6))
        sizes = [s['size'] for s in segments]
        names = [s['name'] for s in segments]
        plt.pie(sizes, labels=names, autopct='%1.1f%%', startangle=90)
        plt.title('User Segment Distribution')
        plt.axis('equal')
        plt.savefig(os.path.join(VISUALIZATION_DIR, 'segment_distribution.png'))
        plt.close()
        
        # 2. Search vs Click Patterns (Scatter Plot)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=user_features,
            x='search_query',
            y='clicked_products',
            hue='cluster',
            palette='deep',
            alpha=0.6
        )
        plt.title('Search vs Click Patterns by Segment')
        plt.xlabel('Number of Searches')
        plt.ylabel('Number of Clicks')
        plt.legend(title='Segment')
        plt.savefig(os.path.join(VISUALIZATION_DIR, 'search_click_patterns.png'))
        plt.close()
        
        # 3. Click Ratio Distribution (Box Plot)
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=user_features,
            x='cluster',
            y='search_click_ratio',
            palette='deep'
        )
        plt.title('Click Ratio Distribution by Segment')
        plt.xlabel('Segment')
        plt.ylabel('Search-to-Click Ratio')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(VISUALIZATION_DIR, 'click_ratio_distribution.png'))
        plt.close()
        
        # 4. Category Distribution (Stacked Bar Chart)
        plt.figure(figsize=(12, 6))
        category_data = []
        for segment in segments:
            for category, count in segment['main_categories'].items():
                category_data.append({
                    'Segment': segment['name'],
                    'Category': category,
                    'Count': count
                })
        category_df = pd.DataFrame(category_data)
        sns.barplot(
            data=category_df,
            x='Segment',
            y='Count',
            hue='Category',
            palette='deep'
        )
        plt.title('Top Categories by Segment')
        plt.xlabel('Segment')
        plt.ylabel('Number of Users')
        plt.xticks(rotation=45)
        plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_DIR, 'category_distribution.png'))
        plt.close()
        
        # 5. Search Frequency Distribution (Histogram)
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=user_features,
            x='search_query',
            hue='cluster',
            multiple='stack',
            palette='deep'
        )
        plt.title('Search Frequency Distribution by Segment')
        plt.xlabel('Number of Searches')
        plt.ylabel('Number of Users')
        plt.legend(title='Segment')
        plt.savefig(os.path.join(VISUALIZATION_DIR, 'search_frequency.png'))
        plt.close()
        
        # 6. Semantic Space Visualization 
        embedding_cols = [col for col in user_features.columns if col.startswith('embedding_')]
        if len(embedding_cols) >= 2:
            plt.figure(figsize=(10, 8))
            for cluster in user_features['cluster'].unique():
                cluster_data = user_features[user_features['cluster'] == cluster]
                plt.scatter(
                    cluster_data[embedding_cols[0]],
                    cluster_data[embedding_cols[1]],
                    label=f'Cluster {cluster}',
                    alpha=0.6
                )
            plt.title('User Segments in Semantic Space')
            plt.xlabel('Embedding Dimension 1')
            plt.ylabel('Embedding Dimension 2')
            plt.legend()
            plt.savefig(os.path.join(VISUALIZATION_DIR, 'semantic_segments.png'))
            plt.close()
        
        logger.info("All visualizations have been generated successfully")
        
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        raise

def generate_segment_report(segments):
    # Generate a detailed report about user segments
    report = []
    report.append("# User Segment Analysis Report")
    report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for segment in segments:
        report.append(f"\n## {segment['name']}")
        report.append(f"- Cluster ID: {segment['cluster']}")
        report.append(f"- Segment Size: {segment['size']} users")
        report.append(f"- Average Searches: {segment['avg_searches']:.1f}")
        report.append(f"- Average Clicks: {segment['avg_clicks']:.1f}")
        report.append(f"- Average Click Ratio: {segment['avg_click_ratio']:.2f}")
        
        report.append("\n**Top Categories:**")
        for category, count in segment.get('main_categories', {}).items():
            report.append(f"- {category}: {count} users")
        
        # Add semantic profile if available
        if 'avg_embedding' in segment:
            report.append("\n**Semantic Profile:**")
            report.append("- Embedding profile: " + 
                         ", ".join([f"PC{i+1}: {val:.2f}" for i, val in enumerate(segment['avg_embedding'])]))
    
    # Write report to file
    with open(os.path.join(VISUALIZATION_DIR, 'segment_report.md'), 'w') as f:
        f.write('\n'.join(report))

def main():
    """Main function to perform user segmentation."""
    try:
        logger.info("Starting enhanced user segmentation analysis...")
        
        # Get user search data
        logger.info("Retrieving user search data...")
        search_data = get_user_search_data()
        if search_data.empty:
            logger.warning("No search data found!")
            return
        
        # Get search embeddings
        logger.info("Generating search embeddings...")
        search_data = get_search_embeddings(search_data)
        
        # Create user features
        logger.info("Creating user features...")
        user_features = create_user_features(search_data)
        
        # Reduce embedding dimensionality
        logger.info("Reducing embedding dimensions...")
        user_features, n_semantic_features = reduce_embedding_dimension(user_features)
        
        # Perform main clustering
        logger.info("Performing main clustering...")
        user_features = perform_clustering(user_features, n_semantic_features)
        
        # Analyze segments
        logger.info("Analyzing segments...")
        segments = analyze_segments(user_features)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        visualize_segments(user_features, segments)
        
        # Generate main segment report
        logger.info("Generating main segment report...")
        generate_segment_report(segments)
        
        # Perform pair-wise clustering analysis
        logger.info("Performing pair-wise clustering analysis...")
        from pair_wise_analysis import main as perform_pair_wise_analysis
        pair_wise_results = perform_pair_wise_analysis(user_features)
        
        logger.info("Analysis completed successfully!")
    except Exception as e:
        logger.exception(f"Critical error in main function: {str(e)}")

if __name__ == "__main__":
    main()