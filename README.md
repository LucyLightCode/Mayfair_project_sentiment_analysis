# Mayfair Sentiment Analysis Project

## Project Overview

This project performs sentiment analysis on a dataset of Mayfair hotel reviews using Natural Language Processing (NLP) techniques. The goal is to classify reviews into positive, negative, and neutral sentiments.

The project involves the following steps:

1. **Data Collection and Preprocessing:**
    - Load the dataset of Mayfair hotel reviews.
    - Clean and preprocess the text data:
        - Remove emojis, URLs, and special characters.
        - Convert text to lowercase.
        - Tokenize the text.
        - Remove stop words.
        - Lemmatize words.
2. **Exploratory Data Analysis (EDA):**
    - Analyze the distribution of ratings.
    - Explore the relationship between review length and ratings.
    - Visualize the average ratings and review count over the years.
    - Identify the top countries with the highest and lowest ratings.
3. **Feature Engineering:**
    - Create a new column named "labels" based on the ratings.
    - Split the dataset into training and testing sets.
    - Apply CountVectorizer and TF-IDF to create numerical representations of the text data.
4. **Model Training and Evaluation:**
    - Train a Multinomial Naive Bayes classifier using both BOW and TF-IDF features.
    - Evaluate the model's performance using accuracy score and classification report.
    - Compare the performance of the model with the VADER sentiment analysis tool.
5. **Deployment:**
    - Create an inference function to predict the sentiment of new reviews.
    - Develop a Flask app for deploying the model.

## Results

- The Bag of Words (BOW) model achieved the highest accuracy, macro F1, and weighted F1 scores, making it the most effective model for this sentiment analysis task.
- The VADER model struggled with the neutral class, resulting in lower macro and weighted scores.
- The TF-IDF model showed decent accuracy but failed to detect the neutral class effectively.

## Conclusion

The BOW model, trained on the cleaned and preprocessed review data, is the recommended model for classifying the sentiment of Mayfair hotel reviews. It provides good overall performance and strikes a balance between all three sentiment classes.

## Future Work

- Improve class balance, particularly for the neutral class.
- Explore other NLP techniques and machine learning models to potentially improve accuracy.
- Fine-tune the model using different hyperparameters.
- Implement a user interface for easier interaction with the model.

## Repository Structure

- `mayfair_sentiment_analysis_11.ipynb`: The main Colab notebook containing the code for data analysis, model training, and evaluation.
- `train_dataset.csv`: The training dataset.
- `test_dataset.csv`: The testing dataset.
- `README.md`: This file providing an overview of the project.

## Dependencies

- Python 3.7+
- Pandas
- NumPy
- Scikit-learn
- NLTK
- TextBlob
- Googletrans
- Langdetect
- Emoji
- Contractions
- Matplotlib
- Seaborn
- Flask

## Usage

1. Clone the repository: `git clone https://github.com/your-username/mayfair-sentiment-analysis.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Colab notebook to reproduce the results.
4. Use the inference function or Flask app to predict the sentiment of new reviews.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

                                     
                                      
                                      ## UPDATED WITH LR, XGBOOST, SVN, RFC, LSTM
This project focuses on performing sentiment analysis on customer reviews of Mayfair hotels, aiming to understand customer satisfaction and identify areas for improvement.

## Dataset

The project uses a combined dataset of customer reviews (`combined_reviews_dataset.csv`), containing the following key columns:

* `REVIEW_CONTENT`: The text of the customer review.
* `RATING`: A numerical rating provided by the customer (1-5).
* `REVIEW_ID`: A unique identifier for each review.
* `DATE`: The date the review was posted.
* `COUNTRY`: The country of origin of the reviewer.

## Methodology

1. **Data Cleaning & Preprocessing:**
   - Remove duplicates and missing values.
   - Language Detection: Keep only English reviews.
   - Text Cleaning: Remove emojis, URLs, special characters, and numbers.
   - Lemmatization: Reduce words to their base form.
   - Stop Word Removal: Filter common words that don't carry significant meaning.
2. **Sentiment Labeling:**
   - Reviews with ratings of 4 or 5 are labeled as "positive."
   - Reviews with ratings of 1 or 2 are labeled as "negative."
   - Reviews with a rating of 3 are labeled as "neutral."
3. **Feature Extraction:**
   - TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text into numerical features.
4. **Model Training & Evaluation:**
   - Various machine learning models are trained and compared, including:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Random Forest
     - XGBoost
     - LSTM (Long Short-Term Memory)
   - Model performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC AUC.

## Results

* **Best Performing Model:** Logistic Regression achieved the highest accuracy and balanced performance across sentiment classes.
* **Challenges:** Accurately predicting neutral sentiment proved to be challenging due to the data imbalance.

## Conclusion

This project provides valuable insights into customer sentiment toward Mayfair hotels. The insights gained can be leveraged to address customer concerns, enhance services, and improve overall customer satisfaction. Further refinements to improve neutral sentiment detection are recommended.

## Future Work

* Experiment with more advanced deep learning models for improved neutral sentiment classification.
* Incorporate aspect-based sentiment analysis to identify specific areas of customer feedback.
* Develop a system for real-time sentiment monitoring and analysis.

## Usage

1. Clone the repository: `git [clone <repository-url>`](https://github.com/LucyLightCode/Mayfair_project_sentiment_analysis.git)
2. Install required libraries: (refer to the notebook for detailed instructions)
3. Run the notebook `mayfair_sentiment_analysis_proj_2.ipynb` in Google Colab.
