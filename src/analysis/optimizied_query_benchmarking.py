import os
import time
from sqlalchemy import create_engine, text
import pandas as pd
import logging
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create report directory if it doesn't exist
REPORT_DIR = os.path.join(os.path.dirname(__file__), 'query_optimization_report')
os.makedirs(REPORT_DIR, exist_ok=True)

# Database configuration
DB_USER = os.getenv('DB_USER', 'itversity_retail_user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'itversity')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME','youcan')

# Create database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Original queries
ORIGINAL_QUERIES = {
    'weekly_active_users': """
    SELECT 
        DATE_TRUNC('week', e.timestamp) as week,
        COUNT(DISTINCT e.user_id) as active_users
    FROM events e
    GROUP BY DATE_TRUNC('week', e.timestamp)
    ORDER BY week;
    """,
    
    'revenue_per_category': """
    SELECT 
        p.category,
        SUM(p.price) as total_revenue
    FROM events e
    JOIN products p ON e.product_id = p.product_id
    WHERE e.event_type = 'purchased'
    GROUP BY p.category
    ORDER BY total_revenue DESC;
    """
}

# Optimized queries
OPTIMIZED_QUERIES = {
        'weekly_active_users': """
        WITH weekly_events AS (
            SELECT 
                DATE_TRUNC('week', timestamp) AS week,
                user_id
            FROM events
        )
        SELECT 
            week,
            COUNT(DISTINCT user_id) AS active_users
        FROM weekly_events
        GROUP BY week
        ORDER BY week;
        """,
        'revenue_per_category': """
        WITH purchased_events AS (
        SELECT product_id
            FROM events
            WHERE event_type = 'purchased'
        )
        SELECT 
            p.category,
            SUM(p.price) AS total_revenue
        FROM purchased_events e
        JOIN products p ON e.product_id = p.product_id
        GROUP BY p.category
        ORDER BY total_revenue DESC;
        """
}

def analyze_query_plan(query: str) -> str:
    """Analyze query execution plan using EXPLAIN ANALYZE."""
    explain_query = f"EXPLAIN {query}"
    with engine.connect() as conn:
        result = conn.execute(text(explain_query))
        plan = "\n".join([row[0] for row in result])
    return plan

def execute_query(query: str) -> Tuple[pd.DataFrame, float, str]:
    start_time = time.time()
    with engine.connect() as conn:
        result = pd.read_sql(text(query), conn)
    execution_time = time.time() - start_time
    
    # Get query execution plan
    execution_plan = analyze_query_plan(query)
    
    return result, execution_time, execution_plan

def benchmark_queries() -> Dict[str, Dict[str, float]]:
    """Benchmark original and optimized queries."""
    results = {}
    
    for query_name in ORIGINAL_QUERIES.keys():
        logger.info(f"Benchmarking {query_name}...")
        
        # Execute original query
        _, original_time, original_plan = execute_query(ORIGINAL_QUERIES[query_name])
        
        # Execute optimized query
        _, optimized_time, optimized_plan = execute_query(OPTIMIZED_QUERIES[query_name])
        
        results[query_name] = {
            'original_time': original_time,
            'optimized_time': optimized_time,
            'improvement': (original_time - optimized_time) / original_time * 100,
            'original_plan': original_plan,
            'optimized_plan': optimized_plan
        }
        
        logger.info(f"Original time: {original_time:.2f}s")
        logger.info(f"Optimized time: {optimized_time:.2f}s")
        logger.info(f"Improvement: {results[query_name]['improvement']:.2f}%")
    
    return results

def plot_benchmark_results(results: Dict[str, Dict[str, float]]):
    """Plot benchmark results."""
    # Prepare data for plotting
    query_names = list(results.keys())
    original_times = [results[q]['original_time'] for q in query_names]
    optimized_times = [results[q]['optimized_time'] for q in query_names]
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Query': query_names * 2,
        'Execution Time (s)': original_times + optimized_times,
        'Version': ['Original'] * len(query_names) + ['Optimized'] * len(query_names)
    })
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Query', y='Execution Time (s)', hue='Version')
    plt.title('Query Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(REPORT_DIR, 'query_benchmark_results.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def main():
    """Main function to run the benchmark and generate report."""
    # Run benchmarks
    results = benchmark_queries()
    
    # Plot results
    plot_path = plot_benchmark_results(results)
    
    # Generate report
    report = "Query Optimization Report\n"
    report += "======================\n\n"
    
    for query_name, metrics in results.items():
        report += f"Query: {query_name}\n"
        report += f"Original execution time: {metrics['original_time']:.2f}s\n"
        report += f"Optimized execution time: {metrics['optimized_time']:.2f}s\n"
        report += f"Performance improvement: {metrics['improvement']:.2f}%\n\n"
        report += "Original Query Plan:\n"
        report += f"{metrics['original_plan']}\n\n"
        report += "Optimized Query Plan:\n"
        report += f"{metrics['optimized_plan']}\n\n"
        report += "----------------------------------------\n\n"
    
    report_path = os.path.join(REPORT_DIR, 'optimizated_query_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Benchmark completed. Results saved to {report_path}")
    logger.info(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main() 