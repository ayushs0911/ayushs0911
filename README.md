# Data Science Portfolio
These are short summaries of my projects, which I've performed to upskill my capabilities as a Data Scientist.<br>

**Table of Contents:**
- [Machine Learning](#machine-learning)<br>
*Tools : Numpy, Pandas, Scikit-Learn, Matplotlib, Seaborn, SQL, Statistics*
- [Computer Vision](#cv)<br>
*Tools : TensorFlow, Pytorch, OpenCV, Resnet34, ViT, HuggingFace Transformers*
- [NLP](#nlp)
- [Time Forecasting](#ts)
- [OpenCV](#ocv)
- [Misc.](#misc)

*****

## Machine Learning <a name="machine-learning"></a>
**[SpaceX Falcon 9 1st stage Landing Prediction](https://github.com/ayushs0911/IBM-Capstone-Project)** <a name="space"></a>
- **Goal of the Project :**
  - SpaceX advertises Falcon 9 rocket launches on its website with a cost of 62 million dollars; other providers cost upward of $165 million per launch, much of the savings is because SpaceX can reuse the first stage. 
  - Therefore if we can determine if the first stage will land, we can determine the cost of a launch. 
- **Highlights:** 
  - Data Collection : HTML Web scrapping from Wikipedia and Request to SpaceX API
  - Data Wrangling and EDA : Connecting to `IBM DB2 Database` and Using `SQL` queries to explore. 
  - Data Viz : Using `Seaborn` and `Matplotlib` library 
  - Algorithms : `Logistic Regression`, `KNN`, `SVM`, `Decision Tree`. 
  - Hyper-Parameter Tuning : Using `GridSearchCV` to find the 'Best Parameters'
  - Decision Trees Performed Best | `Accuracy : ~ 90%`
<br>

**HR Department Case Study: [Employees Attrition Prediction](https://github.com/ayushs0911/Projects/blob/main/HR%20Department:%20Attrition%20Prediction.ipynb)** <a name="employee"></a>
- **Goal of Project :**
  - Perform classification analysis to determine wheather employee will leave the company or not
  - Small Business owners spends 40% of their working hours on tasks that do not generate any income such as hiring. Companies spend 15-20% of employee's salary to recruit new candidate. An average company loses anywhere b/w 1% and 2.5% of their total revenue on time it takes to bring a new hire up to speed. Hiring a new employee costs an average of $7645(0-500 corporation)(Source link text)
- **Highlights :**
  - Imported Libraries, CSV Dataset | Data Cleaning : Nulls, Dropped Un-related columns
  - Data Viz + Satistical Analysis : `Correlation Matrix, Kde Plots, Box Plots, Count Plots` 
  - Performing `ANOVA` And `Chisquare test` for feature selection 
  - OneHot Encoder, Min Max Scaler 
<br>

**Financial Institution Case Study : [Likelihood of approving Loan based on Financial History](https://github.com/ayushs0911/Projects/blob/main/Likelihood_of_approving_a_Loan.ipynb)**

## Computer Vision <a name="cv"></a>
**[Malaria Detection by Blood Sample Images](https://github.com/ayushs0911/Projects/blob/main/Malaria__detection.ipynb)**<a name="malaria"></a>
- **Goal of the Project :** 
  - Detecting whether a Blood Sample is infected by Malarial Parasite. This Model can help in easy detection of malaria cases. 
  - In remote places, where doctors and technicians are not available, this Deep learning model can aid in faster diagnosis and can save lives.
- **The Process :**
  - Imported Libraries and `Tensorflow Datasets`
  - `Data Augmentation` so that our model does not overfits.
  - Data loading through `Batches`, also `prefetching dataset` to make training faster.
  - Also tried `Mixup data augmentation, Cutmix augmentation and Albumenations`  
  - Model construction via `Convolutional Neural Networks` and Used `callbacks`  
  - Plotted Loss Curves, Confusion Matrix
- As it was a Medical diagnosis case, so we have to reduce False Positives (diagosing a person Uninfected, despite being parasatized.)
  - Used `ROC Curve` and Calculated `Threshold` parameter.
  - Then re-plotted the Confusion Matrix


**[Food Vision 101 : Image Classification model using TensorFlow](https://github.com/ayushs0911/Projects/blob/main/Food_Vision_Image_Classificaton_TensorFlow.ipynb)**<a name="food"></a>

## NLP <a name="nlp"></a>
**[Drake Like Lyrics Generation](https://github.com/ayushs0911/Projects/blob/main/NLP/Drake_Lyrics_Generator.ipynb)**<br>
**[Disaster Tweet Prediction : NLP](https://github.com/ayushs0911/Projects/blob/main/Disaster_tweets_Predictor.ipynb)**<br>
**[Sentiment Analysys : Alexa Reviews](https://github.com/ayushs0911/Projects/blob/main/Sentiment_Analysis_Amazon_Alexa.ipynb)**

## Time Forecasting <a name="ts"></a>
**Sales Department Case Study : [Sales Forecasting](https://github.com/ayushs0911/Projects/blob/main/Sales_Forecast_using_Facebook_Prophet.ipynb)**
- **Goal of Project :** 
  - Forecast sales using store, promotion, and competitor data
  - For companies to become competitive and skyrocket their growth, they need to leaverage AI/ML to develop predictive models to forecast sales in future. Predictive models attempt at forceasting future sales based on historical data while taking into account seasonality effects, demand, holidays, promotions, and competition.
- **The Process:**
  - Importing Libraries and Downloading Dataset from Kaggle : Sales Data, Store information Data
  - Data Cleaning : Checked Nulls, Dropped not-important columns,
  - Merged both dataset on 'Store Dataset'
  - Statistical Anlaysis 
  - Used Facebook **Prophet Alogorithm** for Prediction. 

## Open CV <a name="ocv"></a>
- **[Optical Character Recognition](https://github.com/ayushs0911/OpenCV/blob/main/OCR_.ipynb)** : Using OCR with PyTesseract and EASY OCR
- **[Grab Cut Algorithm](https://github.com/ayushs0911/OpenCV/blob/main/GrabCut_Algorithm.ipynb)** : The GrabCut algorithm is particularly useful for applications such as image editing, where it can be used to extract objects from an image for further manipulation or insertion into another image.

## Misc <a name="misc"></a>
- [Google Deep Dream](https://github.com/ayushs0911/DeepDream) : The DeepDream algorithm effective in asking a pre-trained CNN to take a look at the image, identify patterns you recognise and amplify it. It uses representations learned by CNNs to produce these hallucinogenic or 'trippy' images.

