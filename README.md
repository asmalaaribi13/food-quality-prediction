# 🍲🔬 **Food Quality Management and Sensory Analysis Using Machine Learning** 🤖

![GitHub last commit](https://img.shields.io/github/last-commit/asmalaaribi13/food-quality-sensory-analysis)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

This project applies **Machine Learning** to **Food Quality Management** and **Sensory Analysis**. By leveraging clustering models and classification algorithms, the project focuses on analyzing sensory evaluation data to assess food quality, optimize production processes, and classify food products based on various sensory criteria.

The project utilizes datasets related to food quality, sensory evaluations, and production data to understand trends, group products based on sensory attributes, and predict the quality of food products.

---

## 🚀 **Features**

- 📊 **Clustering for Sensory Data**: Use **K-Means** clustering to group food items based on sensory analysis attributes.
- 🧑‍💻 **Classification for Food Quality Prediction**: Build classification models to predict the quality of food products (e.g., **Logistic Regression**, **Random Forest**, **SVM**).
- 📈 **Data Preprocessing**: Handle missing values, scale the data, and prepare it for model training.
- 🔍 **Feature Selection**: Analyze features that contribute most to the classification of food quality and sensory attributes.
- 🌐 **Multiple Datasets**: Use several Excel files to gather different sensory and quality-related data for comprehensive analysis.
- 📊 **Data Visualization**: Visualize results using **matplotlib** and **seaborn** for better insight into food quality and clustering results.

---

## 🧑‍💻 **Technologies Used**

- **Python 3.8+**
- **Jupyter Notebook** (for interactive development)
- **pandas** (data manipulation and preprocessing)
- **scikit-learn** (machine learning algorithms, clustering, classification)
- **matplotlib** and **seaborn** (data visualization)
- **Excel files** (data sources)
  
---

## 📥 **Installation and Setup**

### Prerequisites

Before you begin, ensure that you have the following installed:

- **Python 3.8+** (or higher)
- **pip** (Python package manager)
- **Jupyter Notebook** (for running and interacting with the notebooks)

## 🧑‍💻 **Data Preprocessing and Feature Engineering**

### Handling Missing Data

In any real-world dataset, it is common to have missing values. This project handles missing data by using the following strategies:

- **Imputation**: For numerical features, we use **mean imputation** or **median imputation** depending on the distribution of the data. For categorical features, we fill missing values with the most frequent category.
  
- **Drop Rows/Columns**: In cases where a significant portion of the data is missing or inconsistent, we drop the affected rows or columns.
  
### Feature Scaling

Feature scaling is essential for machine learning models, particularly for algorithms like K-Means clustering, where distance metrics are involved. We use **StandardScaler** to scale the features to have a mean of 0 and a standard deviation of 1. This ensures that all features contribute equally to the model's performance.

### Feature Encoding

For categorical data, we apply **One-Hot Encoding** or **Label Encoding** to convert text-based categories into numerical form that can be processed by machine learning algorithms.

### Feature Selection

Feature selection techniques such as **Correlation Matrix** and **Feature Importance** are used to identify the most relevant features that contribute to the prediction of food quality and the clustering of sensory attributes. Unimportant features are dropped to improve model efficiency and avoid overfitting.

---

## 🤖 **Machine Learning Models**

### Clustering: K-Means

- **K-Means Clustering**: This unsupervised algorithm groups food products into clusters based on sensory attributes (e.g., taste, texture, smell). The goal is to identify patterns in the sensory data and group similar products together. We use the **elbow method** to determine the optimal number of clusters.
  
  - **Steps**:
    1. Load the sensory data.
    2. Preprocess the data by scaling and handling missing values.
    3. Apply K-Means clustering and visualize the clusters using **matplotlib**.
    4. Analyze cluster centers to understand the typical sensory attributes of each group.

### Classification Models

- **Logistic Regression**: Used for binary classification tasks to predict whether a food product meets the required quality standards (high quality vs. low quality).
  
- **Random Forest**: A versatile classifier that builds multiple decision trees and aggregates their results. It is highly effective for handling complex relationships in the data.
  
- **Support Vector Machine (SVM)**: A classifier that aims to find the best hyperplane separating different classes. It is particularly useful for high-dimensional data like sensory attributes.

### Model Evaluation

- **Cross-Validation**: To assess the models’ generalizability, **K-fold cross-validation** is used to ensure that the model performs well on unseen data.
  
- **Performance Metrics**: The models are evaluated using the following metrics:
    - **Accuracy**: The proportion of correctly predicted labels.
    - **Precision**: The proportion of positive predictions that are actually correct.
    - **Recall**: The proportion of actual positives correctly predicted by the model.
    - **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two.

- **Confusion Matrix**: A confusion matrix is used to visualize the model's performance, especially for classification models. It shows the true positives, false positives, true negatives, and false negatives, helping identify areas of improvement.

---

## 🛠️ **Model Deployment (Optional)**

Although not implemented in the current version of the project, model deployment could involve creating a **Flask** or **FastAPI** web application to expose the machine learning models as APIs. These APIs can be used by food manufacturers or quality control departments to make real-time predictions on the quality of food products based on sensory data.

---

## 📊 **Data Visualization**

Data visualization is an essential part of this project to understand the underlying patterns and results. The following types of visualizations are used:

- **Pairplot and Heatmaps**: Visualizing the relationships between different sensory attributes.
- **Cluster Visualization**: Using **matplotlib** and **seaborn**, we plot the results of K-Means clustering to show how food products are grouped based on sensory attributes.
- **Feature Importance**: Visualizing which features contribute most to the classification of food quality using bar charts and feature importance scores from models like **Random Forest**.

---

## 🧑‍💻 **Usage**

After setting up the project and installing the dependencies, you can interact with the Jupyter Notebooks as follows:

1. **Run Clustering Analysis**:
    - Open the `SensoryAnlaysis_Clustering.ipynb` notebook.
    - Follow the steps to preprocess the sensory data, apply K-Means clustering, and analyze the resulting clusters.

2. **Run Classification Models**:
    - Open the `AnimalLab_Classification.ipynb` notebook.
    - Follow the steps to preprocess the food quality data, apply classification models, and evaluate model performance.

3. **Data Exploration**:
    - Explore the data visualizations created using `matplotlib` and `seaborn` to gain insights into the relationships between food quality and sensory attributes.

---

## 📂 **Contributing**

We welcome contributions to improve this project. Here are ways you can contribute:

- **Bug Fixes**: If you find any bugs or issues, feel free to open an issue and submit a pull request to fix them.
  
- **Feature Additions**: If you have an idea for an additional feature that could enhance the project, such as model improvement or further data analysis, please submit a pull request.

- **Documentation**: Improvements to documentation are always welcome, especially if you have suggestions for better clarity or additional information.

---

## 💡 **Inspiration**

This project was inspired by the need to improve food quality management using data science and machine learning techniques. Sensory evaluation is critical for food manufacturers, and by using clustering and classification models, this project aims to optimize food quality control processes and improve food product standards.

---

## 📫 **Contact**

If you have any questions or feedback about the project, feel free to reach out via:

- **Email**: asma.laaribi@outlook.com
- **GitHub**: [asmalaaribi13](https://github.com/asmalaaribi13)

---

## ❤️ **Acknowledgements**

- Thanks to **scikit-learn**, **pandas**, **matplotlib**, and **seaborn** for providing the tools and libraries that made this project possible.
- Special thanks to the contributors and open-source community for their continued support and development of these powerful data science and machine learning libraries.

---

Developed with ❤️ by Asma Laaribi

