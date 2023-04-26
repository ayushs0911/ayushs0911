# Data Science Portfolio
These are various projects which I've performed to upskill my capabilities as a Data Scientist.<br>


## Machine Learning
**[SpaceX Falcon 9 1st stage Landing Prediction](https://github.com/ayushs0911/IBM-Capstone-Project)**
- **Goal of the Project** : 
  - SpaceX advertises Falcon 9 rocket launches on its website with a cost of 62 million dollars; other providers cost upward of $165 million per launch, much of the savings is because SpaceX can reuse the first stage. 
  - Therefore if we can determine if the first stage will land, we can determine the cost of a launch. 
- **The Process:**
  - Data Collection : HTML Web scrapping from Wikipedia and Request to SpaceX API
  - Data Wrangling 
  - Exploratory Data Analysis : Connecting to IBM DB2 Database and Using SQL queries to explore the data. 
  - Data Visualization : Using Seaborn and Matplotlib library 
  - Algorithms : Logistic Regression, KNN, SVM, Decision Tree. 
  - Hyper-Parameter Tuning : Using GridSearchCV to find the 'Best Parameters'
  - Decision Trees Performed Best | Accuracy : ~ 90%
<br>

**HR Department Case Study: [Employees Attrition Prediction](https://github.com/ayushs0911/Projects/blob/main/HR%20Department:%20Attrition%20Prediction.ipynb)** <br>
**Financial Institution Case Study : [Likelihood of approving Loan based on Financial History](https://github.com/ayushs0911/Projects/blob/main/Likelihood_of_approving_a_Loan.ipynb)**

## Computer Vision 
**[Malaria Detection by Blood Sample Images](https://github.com/ayushs0911/Projects/blob/main/Malaria__detection.ipynb)**<br>
- **Goal of the Project :** 
  - Detecting whether a Blood Sample is infected by Malarial Parasite. This Model can help in easy detection of malaria cases. 
  - In remote places, where doctors and technicians are not available, this Deep learning model can aid in faster diagnosis and can save lives.
- **The Process :**
  - Imported Libraries, Downloaded Dataset from Tensorflow Datasets
  - Data Visualisation to see how our images look.
  - Data Augmentation so that our model does not overfits.
  - Data loading through Batches, also prefetching dataset to make training faster.
  - Also tried Mixup data augmentation, Cutmix augmentation and Albumenations but couldn't use it as my processing power was not able to comprehend it.
  - Model construction via Convolutional Neural Networks
  - Introduced callbacks to make training more efficient
  - Plotted Loss Curves, Confusion Matrix
- As it was a Medical diagnosis case, so we have to reduce False Positives (diagosing a person Uninfected, despite being parasatized.)
  - Used ROC Curve and Calculated Threshold parameter.
  - Then re-plotted the Confusion Matrix


**[Food Vision 101 : Image Classification model using TensorFlow](https://github.com/ayushs0911/Projects/blob/main/Food_Vision_Image_Classificaton_TensorFlow.ipynb)**<br>

## NLP
**[Drake Like Lyrics Generation](https://github.com/ayushs0911/Projects/blob/main/NLP/Drake_Lyrics_Generator.ipynb)**<br>
**[Disaster Tweet Prediction : NLP](https://github.com/ayushs0911/Projects/blob/main/Disaster_tweets_Predictor.ipynb)**<br>
**[Sentiment Analysys : Alexa Reviews](https://github.com/ayushs0911/Projects/blob/main/Sentiment_Analysis_Amazon_Alexa.ipynb)**

## Time Forecasting
**Sales Department Case Study : [Sales Forecasting](https://github.com/ayushs0911/Projects/blob/main/Sales_Forecast_using_Facebook_Prophet.ipynb)**

## Open CV
**[Optical Character Recognition](https://github.com/ayushs0911/OpenCV/blob/main/OCR_.ipynb)**
**[Grab Cut Algorithm](https://github.com/ayushs0911/OpenCV/blob/main/GrabCut_Algorithm.ipynb)**

## Misc 
- [Google Deep Dream](https://github.com/ayushs0911/DeepDream)

