### Project Objective: Classification of Anemia Types and Identification of Indicative Factors Using CBC Report Samples

**Objective:**

The primary objective of this project is to develop a robust machine learning model to classify different types of anemia based on Complete Blood Count (CBC) report samples. Additionally, the project aims to identify key hematological factors that are indicative of various anemia types. This will be achieved through comprehensive data analysis, model development, and interpretation of feature importance, ultimately providing insights that can aid in the accurate diagnosis and understanding of anemia.

**Specific Goals:**

1. **Data Exploration and Understanding:**
   - Conduct an exploratory data analysis (EDA) to understand the distribution and characteristics of CBC report samples.
   - Visualize the data to identify patterns and correlations between different blood parameters and anemia types.

2. **Data Preprocessing:**
   - Handle missing values, outliers, and noise in the dataset.
   - Normalize and standardize numerical features to ensure consistency and improve model performance.
   - Encode categorical variables appropriately for inclusion in machine learning models.

3. **Feature Engineering:**
   - Create new features from existing blood parameters if necessary to enhance model performance.
   - Select the most relevant features through feature selection techniques to improve model efficiency and interpretability.

4. **Model Development:**
   - Train multiple classification models (e.g., Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, Support Vector Machines, and Neural Networks) to classify different types of anemia.
   - Evaluate the models using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

5. **Hyperparameter Tuning and Model Optimization:**
   - Optimize model performance by tuning hyperparameters using techniques like grid search and random search.
   - Validate the models using cross-validation to ensure robustness and generalizability.

6. **Model Interpretation and Feature Importance:**
   - Interpret the results of the best-performing model to understand which CBC parameters are most indicative of different types of anemia.
   - Use feature importance scores, SHAP values, or LIME to provide insights into model decisions.

7. **Deployment and Practical Application:**
   - Develop a user-friendly interface for the model, allowing healthcare professionals to input CBC report data and receive anemia type predictions.
   - Test the deployed model with new, unseen data to ensure its applicability in real-world clinical settings.

8. **Reporting and Documentation:**
   - Document the entire project process, including methodology, results, and conclusions.
   - Create visualizations and reports to effectively communicate findings to stakeholders, including healthcare professionals and researchers.

9. **Future Work and Recommendations:**
   - Suggest areas for future research and potential improvements, such as incorporating additional clinical data or using more advanced machine learning techniques.
   - Propose further studies to validate the model across different populations and healthcare settings.

By achieving these goals, the project will provide valuable tools and insights for the classification and understanding of anemia types based on CBC report samples, contributing to better diagnostic practices and patient outcomes.

INTRODUCTION

Anemia is a common blood disorder characterized by a deficiency in the number or quality of red blood cells (RBCs) or hemoglobin, leading to reduced oxygen transport capacity in the blood. It affects millions of people worldwide and can result from various underlying causes, including nutritional deficiencies, chronic diseases, genetic disorders, and bone marrow problems. Accurate classification of anemia types is crucial for effective diagnosis, treatment, and management of the condition.

Complete Blood Count (CBC) tests are one of the most frequently used diagnostic tools in medicine, providing a comprehensive overview of the hematological parameters of an individual. CBC reports typically include measurements such as hemoglobin concentration, hematocrit, RBC count, mean corpuscular volume (MCV), mean corpuscular hemoglobin (MCH), and several other indices that can provide insights into a person's overall health and help identify different types of anemia.

The primary objective of this project is to leverage machine learning techniques to classify different types of anemia based on CBC report samples. Additionally, the project aims to identify key hematological factors that indicate various anemia types, enhancing our understanding of the condition's underlying mechanisms and improving diagnostic accuracy. By integrating advanced data analysis and machine learning models, we aim to develop a robust, accurate, and interpretable tool for anemia classification, which can be utilized by healthcare professionals for better patient outcomes.

This project involves several key steps, including data exploration and visualization, data preprocessing, feature engineering, model development, hyperparameter tuning, and model interpretation. We will employ various classification algorithms, evaluate their performance using appropriate metrics, and select the best-performing model for deployment. Furthermore, we will interpret the results to determine which CBC parameters are most indicative of different anemia types, providing valuable insights for clinical practice.

Ultimately, this project aims to contribute to the field of hematology by developing an effective tool for anemia classification and by providing deeper insights into the factors that drive different types of anemia. Through careful analysis and rigorous model development, we hope to enhance the diagnostic process and support healthcare professionals in delivering more precise and personalized care to patients with anemia.

RESEARCH METHODOLOGY
Mean, Median, Mode
- **Mean**: The average of a set of numbers, calculated by summing all the values and dividing by the count of values. It provides a central value but can be affected by outliers.
- **Median**: The middle value of a dataset when ordered. It is robust to outliers and represents the 50th percentile.
- **Mode**: The most frequently occurring value in a dataset. It is useful for identifying the most common value.

Quantiles
Quantiles are values that divide a dataset into equal-sized, consecutive subsets. Key quantiles include:
- **Quartiles**: Divide data into four equal parts (Q1, Q2/median, Q3).
- **Percentiles**: Divide data into 100 equal parts. The 25th percentile (Q1), 50th percentile (median), and 75th percentile (Q3) are commonly used.

Correlation Matrix
A correlation matrix displays the correlation coefficients between pairs of variables in a dataset. The coefficients range from -1 to 1, indicating the strength and direction of linear relationships:
- **1**: Perfect positive correlation.
- **0**: No correlation.
- **-1**: Perfect negative correlation.

Density Curves
Density curves are smoothed representations of a dataset's distribution, similar to histograms but continuous. They provide insights into the distribution shape, central tendency, and spread of data.

Model Building
Model building involves selecting and training algorithms on data to predict outcomes or understand relationships. Key steps include:
- **Data Preprocessing**: Cleaning and preparing data for analysis.
- **Feature Selection**: Choosing relevant variables.
- **Model Training**: Using algorithms to learn from data.
- **Model Evaluation**: Assessing performance using metrics like accuracy, precision, recall, and F1-score.

Decision Trees
Decision trees are a type of supervised learning algorithm used for classification and regression tasks. They split data into subsets based on feature values, creating a tree-like model of decisions:
- **Root Node**: The top node representing the entire dataset.
- **Internal Nodes**: Decision points based on feature values.
- **Leaf Nodes**: Final output labels or values.

Random Forest
Random forests are an ensemble learning method that combines multiple decision trees to improve prediction accuracy and control overfitting. Each tree is trained on a random subset of the data, and the final prediction is made by averaging (regression) or majority voting (classification):
- **Ensemble Method**: Combines predictions from several models to enhance performance.
- **Robustness**: Less prone to overfitting compared to individual decision trees.
- **Feature Importance**: Provides insights into the importance of variables in making predictions.

Feature Importance Analysis
Feature importance analysis identifies and quantifies the contribution of each feature to a machine learning model's predictive power. Methods include coefficients in linear models, impurity reduction in tree-based methods, permutation importance, and advanced techniques like SHAP and LIME. Understanding feature importance improves model transparency, performance, and provides valuable domain insights. This analysis helps in selecting relevant features and interpreting the model's decision-making process effectively.

By understanding and applying these statistical and machine learning concepts, you can analyze data effectively and build robust predictive models.

ANALYSIS

 WBC              LYMp           NEUTp              LYMn            NEUTn             RBC        
 Min.   : 0.800   Min.   : 6.20   Min.   :   0.70   Min.   : 0.200   Min.   : 0.500   Min.   : 1.360  
 1st Qu.: 6.000   1st Qu.:25.84   1st Qu.:  71.10   1st Qu.: 1.881   1st Qu.: 5.100   1st Qu.: 4.190  
 Median : 7.400   Median :25.84   Median :  77.51   Median : 1.881   Median : 5.141   Median : 4.600  
 Mean   : 7.863   Mean   :25.84   Mean   :  77.51   Mean   : 1.881   Mean   : 5.141   Mean   : 4.708  
 3rd Qu.: 8.680   3rd Qu.:25.84   3rd Qu.:  77.51   3rd Qu.: 1.881   3rd Qu.: 5.141   3rd Qu.: 5.100  
 Max.   :45.700   Max.   :91.40   Max.   :5317.00   Max.   :41.800   Max.   :79.000   Max.   :90.800  
      HGB              HCT               MCV              MCH               MCHC            PLT     
 Min.   :-10.00   Min.   :   2.00   Min.   :-79.30   Min.   :  10.90   Min.   :11.50   Min.   : 10  
 1st Qu.: 10.80   1st Qu.:  39.20   1st Qu.: 81.20   1st Qu.:  25.50   1st Qu.:30.60   1st Qu.:157  
 Median : 12.30   Median :  46.15   Median : 86.60   Median :  27.80   Median :32.00   Median :213  
 Mean   : 12.18   Mean   :  46.15   Mean   : 85.79   Mean   :  32.08   Mean   :31.74   Mean   :230  
 3rd Qu.: 13.50   3rd Qu.:  46.15   3rd Qu.: 90.20   3rd Qu.:  29.60   3rd Qu.:32.90   3rd Qu.:293  
 Max.   : 87.10   Max.   :3715.00   Max.   :990.00   Max.   :3117.00   Max.   :92.80   Max.   :660  
      PDW             PCT           Diagnosis        
 Min.   : 8.40   Min.   : 0.0100   Length:1281       
 1st Qu.:13.30   1st Qu.: 0.1700   Class :character  
 Median :14.31   Median : 0.2603   Mode  :character  
 Mean   :14.31   Mean   : 0.2603                     
 3rd Qu.:14.70   3rd Qu.: 0.2603                     
 Max.   :97.00   Max.   :13.6000    

Interpretation:- **WBC**: Median is 7.4, indicating a typical range of white blood cells.
- **LYMp and NEUTp**: Median percentages of lymphocytes (25.84%) and neutrophils (77.51%) show typical distributions.
- **LYMn and NEUTn**: Medians are 1.881 and 5.141, respectively, reflecting common lymphocyte and neutrophil counts.
- **RBC**: Median of 4.6 indicates normal red blood cell count, but extreme values suggest outliers.
- **HGB**: Median is 12.3, typical for hemoglobin levels, but negative and extremely high values indicate data issues.
- **HCT, MCV, MCH, MCHC**: Medians are within normal ranges, but extreme values again indicate potential outliers.
- **PLT**: Median is 213, showing normal platelet count, with a wide range suggesting variability.
- **PDW and PCT**: Medians are 14.31 and 0.2603, respectively, within expected ranges for platelet distribution and plateletcrit.

1st plot
 Interpretation of Density Plots
These density plots show the distribution of various hematological parameters in the dataset:
1. **WBC (White Blood Cell count)**:
   - Most values cluster around 5-10, indicating a normal range, but there are a few extreme values up to 45.7.
2. **LYMp (Lymphocyte percentage)**:
   - The majority of values are around 25%, with a long tail extending to higher percentages, indicating a skewed distribution.
3. **NEUTp (Neutrophil percentage)**:
   - Most values are around 70-80%, but there are extreme outliers, significantly higher than typical values.
4. **LYMn (Lymphocyte number)**:
   - Most values are around 0-5, showing a typical distribution with some high outliers.
5. **NEUTn (Neutrophil number)**:
   - Values are mostly around 0-10, with some extreme outliers up to 79.
6. **RBC (Red Blood Cell count)**:
   - The distribution peaks around 4-5, indicating a normal RBC count, but has extreme values up to 90.8.
7. **HGB (Hemoglobin)**:
   - The majority of values are around 12-15, but there are some extremely high outliers.
8. **HCT (Hematocrit)**:
   - The main cluster is around 30-50, with extreme outliers indicating potential data entry errors.
9. **MCV (Mean Corpuscular Volume)**:
   - Most values are between 80-100, typical for MCV, but there are very high outliers.
10. **MCH (Mean Corpuscular Hemoglobin)**:
    - The majority of values are around 20-30, with significant outliers.
11. **MCHC (Mean Corpuscular Hemoglobin Concentration)**:
    - Most values are around 30-35, indicating a normal range, but some outliers are very high.
12. **PLT (Platelet count)**:
    - Values cluster around 200-300, with a long tail and some very high outliers.
13. **PDW (Platelet Distribution Width)**:
    - Most values are around 10-20, with a few extreme outliers.
14. **PCT (Plateletcrit)**:
    - The distribution is heavily skewed with most values close to 0.1-0.3, but some extreme values.
### Key Observations

- **Outliers and Extreme Values**: Several parameters (e.g., NEUTp, HCT, MCV) have extreme outliers, which may indicate data entry errors or rare cases.
- **Skewed Distributions**: Some parameters like LYM% and PCT show skewed distributions, suggesting non-normality.
- **Normal Ranges**: Parameters like WBC, RBC, HGB, and PLT mostly fall within expected clinical ranges but still include outliers.


 Here's a summary and interpretation of the key steps and results:
Stepwise Model Selection

    Initial Model:
        You started with a multinomial logistic regression model including all features.
        Used stepwise selection based on AIC to refine the model.

    Final Model:
        The final selected model includes WBC, RBC, HGB, MCV, MCHC, PLT, and PCT.
        AIC of the final model is 1092.2.

Model Summary

    Coefficients: The final model coefficients for each class show the contribution of each predictor.
    Std. Errors: Standard errors for the coefficients indicate the variability of the estimates.
    Residual Deviance and AIC: The residual deviance is 964.223, and the AIC is 1092.223, indicating a good fit.

Confusion Matrix for Predictions

    The confusion matrix shows the performance of the model on the test data.
    Healthy: Correctly classified 84 out of 102 instances.
    Iron Deficiency Anemia: Correctly classified 51 out of 58 instances.
    Leukemia: Correctly classified 7 out of 14 instances.
    The overall performance shows the model is quite accurate in classifying most conditions.

Likelihood Ratio Test

    Conducted a likelihood ratio test comparing the full model to a null model.
    Significance: The test is highly significant (p < 2.2e-16), indicating that the model provides a better fit than the null model.

Random Forest Model

    Trained a random forest model which showed higher accuracy:
        Train Accuracy: 1.0000
        Test Accuracy: 0.9759
    Feature Importance: The random forest model's importance scores reveal the most important features for classification.

Visualization

    Confusion Matrices: Plots for training and testing confusion matrices visualize the performance.
    Feature Importance: Bar plot showing the importance of features from the random forest model.
    Histograms and Density Plots: Distribution of numeric features in the dataset.
    Q-Q Plots: Check for normality of numeric features.
    Correlation Heatmap: Visualizes the correlation between numeric features using Spearman's method.
    Class Distribution: Bar plots showing the count of each diagnosis class in the training set.

Interpretation and Next Steps

    Model Performance: The random forest model outperforms the logistic regression model in terms of accuracy. This suggests that the non-linear relationships captured by the random forest are more effective for this classification task.
    Feature Selection: Important features include RBC, WBC, HGB, and MCV among others. These are critical for distinguishing between different types of anemia and related conditions.
    Visual Analysis: The visualizations provide a comprehensive view of data distribution and feature importance which helps in understanding the model's decisions.
