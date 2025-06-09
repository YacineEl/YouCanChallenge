import os
import logging
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create visualization directory
VISUALIZATION_DIR = os.path.join('src', 'segmentation' ,'visualization')
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def perform_clustering_pair_by_pair(user_features):
    # Perform clustering analysis on pairs of user features to identify patterns and relationships.
    
    # Define feature pairs to analyze - only behavioral features
    feature_pairs = [
        ('search_query', 'clicked_products'),
        ('search_query', 'search_click_ratio'),
        ('clicked_products', 'search_click_ratio')
    ]
    
    results = {}
    
    for feature1, feature2 in feature_pairs:
        try:
            X = user_features[[feature1, feature2]].values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            results[f"{feature1}_vs_{feature2}"] = {
                'clusters': clusters,
                'centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_,
                'feature1': feature1,
                'feature2': feature2,
                'data': X
            }
            
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6)
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                       c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')
            
            plt.title(f'Clustering Analysis: {feature1} vs {feature2}')
            plt.xlabel(feature1)
            plt.ylabel(feature2)
            plt.colorbar(scatter, label='Cluster')
            plt.legend()
            
            plt.savefig(os.path.join(VISUALIZATION_DIR, f'clustering_{feature1}_vs_{feature2}.png'))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error analyzing {feature1} vs {feature2}: {str(e)}")
            continue
    
    return results

def generate_pair_wise_report(results, user_features):
    # Generate a detailed report about the pair-wise clustering analysis.
   
    report = []
    report.append("# Pair-wise Clustering Analysis Report")
    report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for pair_name, result in results.items():
        feature1, feature2 = result['feature1'], result['feature2']
        clusters = result['clusters']
        data = result['data']
        
        report.append(f"\n## {pair_name}")
        report.append(f"Features: {feature1} vs {feature2}")
        report.append(f"Number of clusters: 3")
        report.append(f"Clustering inertia: {result['inertia']:.2f}")
        
        # Cluster statistics
        for cluster in range(3):
            cluster_data = data[clusters == cluster]
            report.append(f"\n### Cluster {cluster}")
            report.append(f"Size: {len(cluster_data)} users")
            report.append(f"Mean {feature1}: {cluster_data[:, 0].mean():.2f}")
            report.append(f"Mean {feature2}: {cluster_data[:, 1].mean():.2f}")
            report.append(f"Standard Deviation {feature1}: {cluster_data[:, 0].std():.2f}")
            report.append(f"Standard Deviation {feature2}: {cluster_data[:, 1].std():.2f}")
            
            # Add category distribution for this cluster
            cluster_users = user_features[clusters == cluster]
            if 'main_category' in user_features.columns:
                category_dist = cluster_users['main_category'].value_counts().head(3)
                report.append("\nTop Categories:")
                for category, count in category_dist.items():
                    report.append(f"- {category}: {count} users")
    
    report_path = os.path.join(VISUALIZATION_DIR, 'pair_wise_analysis_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Pair-wise analysis report saved to: {report_path}")
    return report

def main(user_features):
    # Main function to perform pair-wise clustering analysis.
    
    try:
        logger.info("Starting pair-wise clustering analysis...")
        
        results = perform_clustering_pair_by_pair(user_features)
        
        report = generate_pair_wise_report(results, user_features)
        
        logger.info("Pair-wise analysis completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Error in pair-wise analysis: {str(e)}")
        raise

if __name__ == "__main__":
    pass 