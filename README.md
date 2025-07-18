# **Movie Recommendation System**

This project aims to build a personalized movie recommendation system using the **MovieLens dataset**. It combines **collaborative filtering** and **content-based filtering** to produce movie recommendations for users. The system is designed to handle both regular users and new users with little or no rating history.

----

## **Problem Statement** 

Users on the StreamTimeNow platform often struggle to discover content that matches their preferences due to the vast number of available options. With thousands of movies to choose from, users are often overwhelmed and find it difficult to locate movies that match their interests, especially since most don't scroll deeply or explore the site extensively. This results in user-frustration, decision fatigue and in some cases platform abandonment. 

As a result, StreamTimeNow faces a critical challenge in retaining users and maintaining long-term engagement, which directly impacts business sustainability. 

---

## **Project Goal**

The goal of this project is to develop a personalized movie recommendation system that suggests relevant movies to users based on their preferences in order to improve user satisfaction and engagement.

---

## **Project Objectives**

1. Identify the most popular movies based on metrics like number of ratings and average rating
2. Investigate how users rate movies 
3. Developing and evaluating advanced recommender models (e.g., collaborative filtering, matrix factorization, or hybrid methods).
4. Explaining recommendation results, using example user profiles and insights derived from the model.

---

## **Dataset Summary**

- Dataset: [`ml-latest-small`](https://grouplens.org/datasets/movielens/) 
- 100,836 ratings | 610 users | 9,742 movies  
- Files used:
  - `movies.csv` – movie titles + genres
  - `ratings.csv` – user-movie ratings
  - `tags.csv` – user-added tags for describing movies

---
## **Exploratory Data Analysis (EDA)**
Before building the recommendation models, we performed extensive EDA to understand the dataset better. Key insights included:
### - ***Distribution of movie ratings***
<img width="913" height="562" alt="download" src="https://github.com/user-attachments/assets/a1e47122-bb82-4bb9-84a0-b23afb4435b2" />


This histogram visualizes how users rate movies on our platform, showing a clear tendency towards positive feedback with significant peaks at higher rating values.

### - ***Most popular movies and genres***
<img width="1800" height="966" alt="download" src="https://github.com/user-attachments/assets/16739ff4-aae2-4201-b424-b36e305bf77d" />


This bar chart highlights the movies that have received the most ratings in the dataset. It provides insight into the most popular content among our user base

### - ***User rating distribution***

<img width="1781" height="882" alt="download" src="https://github.com/user-attachments/assets/c69f656c-32bd-44e8-a5b0-b20c8a49a965" />


This histogram illustrates the distribution of rating activity across our users. It clearly shows that many users have provided only a limited number of ratings, highlighting the significant 'cold-start user' challenge that our hybrid recommendation system is designed to address


## **Modeling Approach**

 Recommendation strategies that were implemented include:

- **Collaborative Filtering (CF)**: Using neighborhood-based methods (KNN Basic, KNNWithMeans) and matrix factorization (SVD) to capture latent patterns in user-item ratings.
- **Content-Based Filtering (CBF)**: Using TF-IDF vectorization of genres and tags to recommend similar movies.
- **Hybrid Approach**: A weighted and a switching mechanism that combines collaborative and content-based filtering.

Models were evaluated using **RMSE** on a held-out test set and through cross-validation.


## **Evaluation Metrics**

We used RMSE (Root Mean Squared Error) to compare model performance.

| Model                | Average RMSE   |
|---------------------|--------|
| KNN Basic           | 0.9730   |
| KNNWithMeans        | 0.9005   |
| SVD (Collaborative)  | 0.8746   |

The **SVD model** yielded the best results among the collaborative filtering models, and the hybrid approaches leverage its strengths.


## **Business Recommendations**

1. ***Cold-Start Strategy for New Users***  
   Use content-based filtering to recommend movies to new users based on selected genres or movie preferences during onboarding. This mitigates cold-start issues and improves first-time user satisfaction.

2. ***Hybrid Approach for Balanced Recommendations***  
   Implement a weighted or switching hybrid model to combine the strengths of collaborative filtering and content-based filtering, improving recommendation diversity and accuracy.

3. ***Promote Hidden Gems to Maximize Catalog Value***  
   Leverage the model insights to surface highly rated but under-watched movies and push them to relevant users. This increases exposure for lesser-known content and improves catalog utilization across the platform.

4. ***Improve Tag Quality to Enhance Discovery***
Encourage more user tagging and curate high-quality, relevant tags to enrich movie metadata. This can improve the discoverability of niche content and increase recommendation precision, especially for long-tail or lesser-known titles.


## **Conclusion**
This project demonstrates how a hybrid recommender system can address the content discovery problem on a platform like StreamTimeNow. By combining collaborative filtering and content-based techniques, we’re able to provide high-quality, personalized movie recommendations even for new users.



## **For More Information**
Please review our full analysis in the Notebook above or the presentation.

For any additional questions, please contact:


- **Samwel Kipkemboi** – [samkemboi201@gmail.com](https://mail.google.com/mail/?view=cm&fs=1&to=samkemboi201@gmail.com)  


## **Project Structure**
```
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   └── links.csv
├── data_utils.py
├── .gitignore
├── notebook.ipynb
├── Presentation.pdf
└── README.md

```


