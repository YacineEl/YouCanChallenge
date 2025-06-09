Project Structure:
src/
├── analysis/ ------------------------------------------------- Part I: Optimizing the sql querie + indexing
│   ├── query_benchmarking.py----------------------------------------This will be run twice to compare the performance of indexing
│   ├── optimized_query_benchmarking.py------------------------------Contains the query vs optimizated_query
│   └── report/------------------------------------------------------ Analyze the performance of indexing
│       ├── query_plan_analysis.txt
|       ├── no_indexing_optimization_report.txt
|       ├── indexed_optimization_report.txt
│   └── query_optimization_report/---------------------------------- Analyze the performance of query optimization
│       ├── query_analysis.txt
│       └── optimized_query_report.txt
├── cohort-analysis/ ------------------------------------------ Part II : Cohort Analysis
│   ├── cohort-analysis.ipynb--------------------------------------- Notebook
│   └── cohort_analysis_report/------------------------------------- Report
│       └── cohort_analysis_report.txt
├── data/------------------------------------------------------ Part 0.1 : DB population
│   ├── generate_simple_data.py
│   ├── generate_data.py
│   ├── updated_generate_data.py
│   └── indexing.py
├── database/-------------------------------------------------- Part 0.0: DB setup
│   └── setup.py
├── segmentation/---------------------------------------------- Part III: Segmentation
│   ├── user_segmentation.py
│   ├── pair_wise_analysis.py
│   ├── setup_elasticsearch.py
│   └── visualization/
│       ├── clustering_{feature1}_vs_{feature2}.png
│       ├── pair_wise_analysis_report.md
        └── segment_report.md


-----------------------------------------------------------------------------


1 - Create Tables [No Index for now]: .\src\database\setup.py
2 - Populate db : .\src\data\generate_simple_data.py


Part I: Data Exploration & SQL Query Optimization

1 - Explore the preformance of the sql quries before indexing: 

    -> run the script: .\src\analysis\query_benchmarking.py
        . The scripts will output: 
            . Report of execution plan fro both query:  'no_indexing_optimization_report.txt'

    -> The file .\src\analysis\report\query_plan_analysis.txt will contain an expolarion of the approach of the optimization:
        . Bottlenecks + Indexes that should be added

2 - Explore the preformance of the sql quries after indexing: 

    -> run the script: .\src\data\indexing.py
        . This script will add the indexes mentionned in the query_plan_analysis.txt file. 

    -> rerun the script: .\src\analysis\query_benchmarking.py [modified the name of output files] 
        . The scripts will output: 
            . Report of execution plan fro both query 'indexed_optimization_report.txt'

    -> return to the file .\src\analysis\report\query_plan_analysis.txt for the analysis of the results

    -> The file .\src\analysis\report\query_plan_analysis.txt will contain an expolarion of the optimization preformance. 

3 - Explore the preformance optimized sql quries: 

    -> The file src\analysis\query_optimization_report\query_analysis.txt will contain an analysis of the optimization steps.

    -> run the script: src\analysis\optimizied_query_benchmarking.py
        . This script will output a report on the performances on the queries: src\analysis\query_optimization_report\optimizated_query_report.txt



Part II: Cohort Analysis

1 - This section is straightforward, the only issue was to update the approach in the genaration of data. 

    -> run file 'src\data\generate_simple_data.py'; [it will drop the rows in tables if exists]

    -> run the notebook 'src\cohort-analysis\cohort-analysis.ipynb'
        . This will generate a report:  'src\cohort-analysis\cohort_analysis_report\cohort_analysis_report.txt'. 

ps; there isnt much to go on here since the data is generated in a straightforward manner where:
- All Products are created in the first day 
- Users are created sequentially with their events
- Each user has a random event frequency pattern
- Events are generated week by week with increasing churn probability
- Events are spread across the full 24-hour period of each day


Part III: User Segmentation Analysis

0 - Environment Setup:

    a- Docker Setup:
    -> run 'docker-compose up -d' to start:
        . Elasticsearch service

    b- Elasticsearch Setup:
    -> run the script: 'src\segmentation\setup_elasticsearch.py'
        . Creates 'user_searches' index with mappings:
            - user_id (keyword)
            - search_query (text, standard analyzer)
            - clicked_product_ids (keyword)
            - timestamp (date)
            - category (keyword)
        . Generates sample search data:
            - 1000 users
            - 10 searches per user
            - Random categories and search terms
            - Timestamps within last 30 days

1 - Pair-wise Clustering Analysis:
    -> run the script: 'src\segmentation\user_segmentation.py'
        . This will perform pair-wise clustering on behavioral features:
            - Search volume vs. Click volume
            - Search volume vs. Click ratio
            - Click volume vs. Click ratio
        . Outputs:
            - Visualizations in 'src\visualization\clustering_{feature1}_vs_{feature2}.png'
            - Report in 'src\visualization\pair_wise_analysis_report.md'

2 - Main Clustering Analysis:
    -> The script will also perform main clustering analysis:
        . Features used:
            - Behavioral metrics (searches, clicks, ratios)
            - Category preferences
        . Outputs:
            - Segment visualizations in 'src\visualization\'
            - Main segment report in 'src\visualization\segment_report.md'

3 - Analysis Results:
    . The analysis focuses on:
        - User behavior patterns
        - Search and click relationships
        - Category preferences
        - Segment characteristics and sizes
    . Reports include:
        - Cluster statistics
        - Feature distributions
        - Category preferences
        - Visual representations of segments