# Navigating-Passenger-Contentment-Anticipating-Satisfaction-and-Crafting-Effective-Airline-Surveys
This project aims to revolutionize the way airlines collect and analyze customer feedback by reducing the number of survey questions while maintaining the depth of insights required for informed decision-making.

![Scikit-learn](https://img.shields.io/badge/Scikit-learn-FF0000?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]([https://github.com/Pr0-C0der/exoplanet-detection/blob/main/LICENSE](https://github.com/crosskey12/Navigating-Passenger-Contentment-Anticipating-Satisfaction-and-Crafting-Effective-Airline-Surveys/blob/master/LICENSE))

[![Linkedin Badge](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/aditya-nagulpelli/)](https://www.linkedin.com/in/prathamesh-gadekar-b7352b245/)

## Table of Contents

- [Introduction](#introduction)
- [Business Problem & Objectives](#business-problem-and-objectives)
- [Approach](#approach)
- [Benefits](#benefits)
- [Dataset Description](#dataset-description)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Modelling](#modelling)
- [Results](#results)
- [Conclusion](#conclusion)

# Introduction

In the competitive landscape of the airline industry, understanding and improving customer satisfaction is paramount. Traditional surveys with 20+ questions often result in low participant engagement and limited actionable insights. This project aims to revolutionize the way airlines collect and analyze customer feedback by reducing the number of survey questions while maintaining the depth of insights required for informed decision-making.

# Business Problem and Objectives

- Business Problem:
  The prevalent issue lies in the lengthy and tedious nature of customer feedback surveys. Respondents often find it 
  cumbersome to answer numerous questions, leading to disengagement and potentially inaccurate responses. Despite the 
  importance of gathering feedback, lengthy surveys hinder participation and hinder the overall objective of improving 
  customer experience.
- Objectives:
  The primary goal of this project is to enhance user experience and improve the efficiency of customer feedback analysis 
  for airline companies. By optimizing the survey process, the project aims to achieve the following:
  - Engagement Improvement: Design a concise and user-friendly feedback survey that encourages higher response rates and 
    more accurate insights.
  - Key Insights Extraction: Identify and retain the most critical questions that effectively capture customer sentiment, 
    allowing the airline to make informed decisions based on actionable insights.
  - Efficient Analysis: Develop a robust analysis framework that prioritizes and evaluates the impact of various parameters 
    on customer satisfaction, enabling the airline to allocate resources effectively for improvement.

# Approach

The project will follow a systematic approach to achieve its objectives:
- Data Collection: Collect survey data of a airline asking maximum number of question having diverse data.
- Analyze data: Analyze data to look what whats good and whats bad with the airline, understand customer prefferences and behaviour
- Machine Learning Framework: Develop a sophisticated Machine Learning framework that identifies correlations between survey responses and customer satisfaction. Utilize machine learning algorithms to uncover hidden patterns and factors driving satisfaction.
- Parameter Prioritization: Apply statistical methods and predictive modeling to prioritize factors that have the most significant impact on customer satisfaction. This aids the airline in making informed strategic decisions.

# Benefits

The successful implementation of this project offers numerous benefits to the airline company:
- Enhanced Customer Experience: Streamlined surveys and prompt feedback mechanisms result in improved customer satisfaction, loyalty, and retention.
- Operational Efficiency: Focusing on key parameters allows the airline to allocate resources efficiently, targeting areas that have the most significant impact on satisfaction.
- Competitive Advantage: By offering a more user-friendly feedback process, the airline gains a competitive edge in the market, showcasing its commitment to customer-centric improvements.
- Data-Driven Decision Making: Informed decisions based on data-driven insights lead to better strategic planning and resource allocation.

# Dataset Description
The dataset for the following project was collected by a US airline survey.It is divided into training and testing dataset

- Trainset:
  - 104000 rows or observations with 25 features.
- Testset:
  - 26000 rows or observations with 25 features.

## Performance Metric:
In the evaluation of our binary classification model, we have employed recall, precision, and the F1-score as the key performance metrics, with a specific focus on their relevance in the domain of customer satisfaction prediction. Given the nature of predicting customer satisfaction and the practical implications of our model's decisions, understanding the implications of these metrics is of utmost importance.

Recall and precision are pivotal in assessing the effectiveness of our model in the context of customer satisfaction prediction. In this scenario, high recall carries the significance of correctly identifying as many satisfied customers as possible within the dataset. Ensuring that satisfied customers are accurately identified helps in building strong customer relationships, enhancing loyalty, and gaining insights into factors that contribute to satisfaction.

High precision, on the other hand, holds value in minimizing false positives—instances where the model erroneously predicts a customer as satisfied when they are not. This is critical as it avoids unnecessary resource allocation towards retaining customers who may not be at risk of churning. Precision plays a role in resource optimization and cost-effective customer management strategies.

The F1-score acts as a bridge between recall and precision, offering a comprehensive perspective on the model's overall performance. Balancing the trade-off between recall and precision, the F1-score is particularly relevant in customer satisfaction prediction. Striving for an optimal F1-score ensures that our model not only identifies satisfied customers accurately but also minimizes the risk of incorrectly labeling dissatisfied customers as satisfied.

The dynamic interplay of recall, precision, and the F1-score underscores the complexity of customer satisfaction prediction. Achieving a delicate equilibrium among these metrics is essential for predicting customer satisfaction with accuracy and reliability.

# Exploratory Data Analysis

- Customers between age 40-60 show high loyalty rate and often purchase buissness class for buisness travel.
- Customers not between age 40-60 show 20% low loyalty rate and are more likely to purchase Eco class,
-  Satisfied customers of both class are mostlt travelling for buisness(96%) and 70-80+% purchase buisness class
- Unsatisfied customers of both class show high personal travel(50%) and ^60%+ Eco class purchases
- 40-60 satified customers were 98% loyal customers and 96% travelled for buisness
- More than 75% customers who took long flight were neutral or happy about Online Boarding,seat comfort,Inflight entertainment,On-Board service,leg room service,baggage handling,checkin service,inflight service and cleanliness.However,such trend is only seen with seat comfort and baggaeg handling for customers who preffered short flight.
- People who took short flight were generally less happy with Online Boarding,Inflight entertainment,On-Board service,leg room service,checkin service,inflight service and cleanliness.
- Most the the people taking flights >1500 are loyal customers(96%) and purchase business class(83%).
- While people with less than 1500 flight distance shoe 21% drop in loyalty count and 59% purchase eco class.
- For 1500+ Personal travel and Eco class purchases led to unsatisfied customers i.e. very few people for 1500+ were satisfied with eco class.
- Amount of personal travel and Eco classs purchases are consistently high for unsatisfied customers, like 71% of unsatisfied customers for -1500 purchased Eco class.
 - 95%+ customers in 40-60 and 1500+ are loyal customers.
- 80% percent customers in this segment were neutral or dissatisfied with Eco and Eco Plus class.
- Almost all the customers 40-60,1500+ and satisfied were (100%)loyal customers travelling for buisness reason and purchasing Business

# Data Preprocessing
In the data pre-processing phase, several steps are taken to prepare the dataset.

- Data Cleaning - We have replaced null values in "Departure Delay in Minutes" by values in "Arrival Delay in Minutes" as it was found that these two features have linear correlation.
- Pre processing - We have replaced categorical data with numerical data and dropped 2 unwanted index,id columns.

# Modelling 
In this project, we explore various Machine Learning models to to accurately predict the right questions.

We have compared 10 base models and found that Random Forest had the best accuracy.Hence, we used RF to perform forward Feature Selection using mlxtend(intel ml library) function to reduce number of columns from 22 to 10. We saw that the performance had drop a little so we explored performance of various ensembles and chose Stacking Classifier with RF and SVM as base estimator and LR as final estimator.
(Note - Hypertuning was performed using gridsearchcv but default parameters we found to be best (Not for SVM))

# Results
The results of the project are as follows:

Classification report with 22 columns:

Classification report with 10 columns(Stacking Classifier):

# Conclusion

We have successfully helped airline eliminate questions from survey (thereby enchancing user experience by reducing the lengthy and tedious nature of customer feedback surveys) without affecting metrics for passenger satisfaction level prediction
