# Customer Spending Segmentation using K-Means Clustering

This project uses K-Means clustering to segment customers based on their spending habits, transaction frequency, and category preferences. The goal is to uncover patterns in consumer behavior, enabling businesses to tailor their marketing efforts and optimize strategies for different customer segments.

## Project Overview

The project analyzes transaction data, categorizing customer spending behavior into distinct clusters. Using the Elbow Method, we identified the optimal number of clusters (4), and applied K-Means clustering to segment customers. The results provide valuable insights into customer spending, which can help businesses optimize marketing, inventory management, and pricing strategies.

### Key Steps in the Analysis:

1. **Data Preprocessing**: 
   - Standardized the features to ensure that all variables contribute equally to the analysis.
   - Applied Principal Component Analysis (PCA) to reduce dimensionality and simplify visualization while retaining key patterns in the data.

2. **Optimal Number of Clusters**:
   - Used the Elbow Method to determine the optimal number of clusters, identifying 4 as the ideal number.
   
3. **K-Means Clustering**:
   - Applied K-Means clustering to group customers into 4 distinct segments, with each cluster representing a specific set of behaviors.

4. **Cluster Insights**:
   - Cluster 0: High spenders with a preference for Groceries and Personal Hygiene.
   - Cluster 1: Moderate spenders, focusing on Gifts and Groceries.
   - Cluster 2: Balanced spenders across various categories.
   - Cluster 3: Primarily focused on Shopping and Fitness.

5. **Visualization**:
   - Visualized the clusters in 2D using PCA to display customer distribution and cluster centroids.

## Features Engineered:

- **Recency**: Time since each customer’s last transaction.
- **Transaction Frequency**: Number of purchases made by each customer.
- **Average Spending**: Average amount spent by a customer per transaction.
- **Categorical Features**: Payment Method and Location (one-hot encoded).
- **Temporal Features**: Hour of transaction and Weekday of the transaction.

## Future Work

1. **Enhance Clustering**: Incorporate additional features, such as customer demographics or feedback, and explore alternative clustering algorithms like DBSCAN to capture more complex customer behavior patterns.
2. **Refine Marketing Strategies**: Validate the clusters using external metrics (e.g., Silhouette Score) and develop dynamic, time-sensitive clustering models for personalized, evolving marketing strategies.

## Dataset

The dataset used in this project is sourced from Kaggle and contains 10,000 transaction records from 200 unique customers, detailing their spending behavior across various categories like Groceries, Shopping, Fitness, Medical/Dental, Travel, etc.

[Download Dataset](https://www.kaggle.com/datasets/ahmedmohamed2003/spending-habits?resource=download)

## Technologies Used

- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Jupyter Notebook
- K-Means Clustering
- Principal Component Analysis (PCA)
- Elbow Method for Optimal Clusters

## Contact

- **Isabella Lo** | [GitHub](https://github.com/beanbean9339) | [LinkedIn](https://linkedin.com/in/igwlo)
