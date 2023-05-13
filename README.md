Welcome to my portfolio! Here, you'll find brief descriptions of the projects I've undertaken to enhance my skills as a Data Scientist.

**Table of Contents:**
- [Machine Learning](#machine-learning) <br>
  - [Regression Analysis](#ra)<br>
  - [Classification Analysis](#ca)<br>


  *Tools : Numpy, Pandas, Scikit-Learn, Matplotlib, Seaborn, SQL, Statistics*<br>
- [Computer Vision](#cv)<br>
  - [Object Detection](#object)
  - [Image Classification](#ic) <br>

  *Tools : TensorFlow, Pytorch, OpenCV, Albumentations, HuggingFace Transformers*
- [NLP](#nlp)<br>

  *Tools : Spacy, Nltk, Embeddings, LSTM, RNN*
- [Time Forecasting](#ts)
- [Misc.](#misc)

*****

# Machine Learning <a name="machine-learning"></a>
## Regression Analysis <a name = "ra"></a> <br>

**[Body Fat Estimation Flask Web Application](https://github.com/ayushs0911/Regression-Models/tree/main/Body%20Fat%20Estimation)**
- **Goal of Project :** Illustrate multiple regression techniques, for Accurate measurement of body fat as majority of tools  are inconvenient/costly and it is desirable to have easy methods. 
- **Highlights :**
  - `Kde Plots`, `Histogram` `Probability plot` `Boxplot`
  - Checking **outliers** with upper and limit equals to -3 and 3 in normal distribution. 
  - Feature Selection : `Extra Tree Regressor` `Mutual Information Gain` `Variance inflation factor`
  - Result : `Random Forest Regression` performs best with `R2 Score : 99%` 
  - Hosted Website on `FLASK` framework. 


## Classification Analysis <a name = "ca"></a>
**[SpaceX Falcon 9 1st stage Landing Prediction](https://github.com/ayushs0911/IBM-Capstone-Project)** <a name="space"></a>
- **Goal of the Project :** SpaceX advertises Falcon 9 rocket launches on its website with a cost of 62 million dollars; other providers cost upward of $165 million per launch, much of the savings is because SpaceX can reuse the first stage. Therefore if we can determine if the first stage will land, we can determine the cost of a launch. 
- **Highlights:** 
  - HTML `Web scrapping` from Wikipedia and Request to SpaceX API
  - Connecting to `IBM DB2 Database` and Using `SQL` queries to explore. 
  - Using `Seaborn` and `Matplotlib` library 
  - Algorithms : `Logistic Regression`, `KNN`, `SVM`, `Decision Tree`. 
  - Hyper-Parameter Tuning : Using `GridSearchCV`  | Decision Trees Performed Best | `Accuracy : ~ 90%`
<br>

**HR Department Case Study: [Employees Attrition Prediction](https://github.com/ayushs0911/Projects/blob/main/HR%20Department:%20Attrition%20Prediction.ipynb)** <a name="employee"></a>
- **Goal of Project :** Perform classification analysis to determine wheather employee will leave the company or not. Small Business owners spends 40% of their working hours on tasks that do not generate any income such as hiring. Companies spend 15-20% of employee's salary to recruit new candidate. 
- **Goal of Project :**
- **Highlights :**
  - Imported Libraries, CSV Dataset | Data Cleaning : Nulls, Dropped Un-related columns
  - Data Viz + Satistical Analysis : `Correlation Matrix, Kde Plots, Box Plots, Count Plots` 
  - Performing `ANOVA` And `Chisquare test` for feature selection 
  - OneHot Encoder, Min Max Scaler 
<br>

**Financial Institution Case Study : [Likelihood of approving Loan based on Financial History](https://github.com/ayushs0911/Projects/blob/main/Likelihood_of_approving_a_Loan.ipynb)**

# Computer Vision <a name="cv"></a>
## Object Detection <a name="object"></a>
**[Mask Detection using Detectron2 Library](https://github.com/ayushs0911/Object-Detection/tree/main/Mask%20Detection%20using%20Detectron%202)**
- **Goal of Project :** Create an application which can detect wheather a person is wearing mask, not wearing it propoperly and not wearing. 
- **Highlights**
  - Use of `Detectron 2` Library developed by `Facebook AI Research`
  - Used `faster_rcnn_R_50_FPN_3x` for object detection 
  - Trained for `1000 Iterations`

[**Arthropod Taxonomy Orders Object Detection**](https://github.com/ayushs0911/Object-Detection/tree/main/Anthropods%20Object%20Detection)
- **Goal of Project :** Create an object detection model that can accurately and efficiently detect objects in an image or video stream in real-time. 
- **Highlights**
  - Use of `Yolov8` developed by `Ultralytics`
  - Dataset : [Arthropod Taxonomy Orders Object Detection Dataset](https://www.kaggle.com/datasets/mistag/arthropod-taxonomy-orders-object-detection-dataset) | Total of 15,000 Images | Images spread across 7 classes 
  - Exported model to `ONXX` format 

**[YOLO from Scratch](https://github.com/ayushs0911/Object-Detection/tree/main/YOLO%20From%20Scratch%20)**
- **Goal of Project :** Create an object detection model that can accurately and efficiently detect objects in an image or video stream in real-time. 
- **Highlights :**
  - Parsed `XML` which contains annotation of training images with object detection information. 
  - `Albumentations` Library
  - Used `EfficientNet1` and changed top two layers to CNN layers, instead of using Fully connected layers. 
  - Defined `Intersection Over Union` function to measure the overlap between two sets of bounding boxes.
  - Defined `YOLO Loss function`
  - Trained for `50 Epochs` | Results displayed on Validation Dataset. 


## Image Classification<a name="ic"></a>
**[Emotion Detection](https://github.com/ayushs0911/Projects/tree/main/Emotions%20Detection)**
-  **Goal of the Project :** Develop a deep learning model that accurately recognizes emotions from facial expressions for potential applications in psychology and marketing.
- **Highlights:**
  -  Dataset : ~30,000 Training Images, belonging to 7 different Classes
  - `Data Augmentation`, 
  - `Lenet`, `ResNet34`, Transfer Learning `EfficientNet`, FineTuning EfficientNet, Vision Transformer, Using `HuggingFace Transformer`
  - HuggingFace downloaded Model performed best : `Accuracy : ~70%`

**[Malaria Detection by Blood Sample Images](https://github.com/ayushs0911/Projects/blob/main/Malaria__detection.ipynb)**<a name="malaria"></a>
- **Goal of the Project :** Detecting whether a Blood Sample is infected by Malarial Parasite. This Model can help in easy detection of malaria cases. In remote places, where doctors and technicians are not available, this Deep learning model can aid in faster diagnosis and can save lives.
- **Highlights:**
  - `prefetching dataset` to make training faster |`Mixup data augmentation, Cutmix augmentation and Albumenations`  
  - `CNN` and Used `callbacks`  | Plotted Loss Curves, Confusion Matrix
  - `Accuracy : 93%`
- As it was a Medical diagnosis case, so we have to reduce **False Positives** (diagosing a person Uninfected, despite being parasatized.)
  - Used `ROC Curve` and Calculated `Threshold` parameter.
  - Then re-plotted the Confusion Matrix


**[Food Vision 101 : Image Classification model using TensorFlow](https://github.com/ayushs0911/Projects/blob/main/Food_Vision_Image_Classificaton_TensorFlow.ipynb)**<a name="food"></a>


# NLP <a name="nlp"></a>

**[Sentiment Analysys : Alexa Reviews](https://github.com/ayushs0911/Projects/blob/main/Sentiment_Analysis_Amazon_Alexa.ipynb)**<br>
- **Goal of Project :** Based on reviews, predicting whether customers are satisfied with the product or not.
  - Dataset consists of  ~ 3000 Amazon customer reviews (input text), star ratings, date of review, variant and feedback of various amazon Alexa products like Alexa Echo, Echo dots, Alexa Firesticks etc.
- **Highlights :**
  - Data Evaluation, `WordCloud`, Cleaning : droppin not important columns, remove punctuations, 
  - Baseline Model : `TfidfVectorizer`, `MultinomialNB`
  - Model 1 : Conv1D with token Embeddings : `layers.Embedding`, 
  - Model 2 : Feature extraction with pretrained token embeddings : `hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")`
  - Model 3 : Conv1D with character embeddings
  - `custom_token_embed_conv1d`, performs slightlty better than other models.
    - Accuracy : 94.179894	
    - Precision : 0.927576	
    - Recall : 0.941799	
    - F1 Score : 0.921988   


**[Drake Like Lyrics Generation](https://github.com/ayushs0911/Projects/blob/main/NLP/Drake_Lyrics_Generator.ipynb)**<br>
- **Goal of Project :** Text Generation model, which outputs Drake Style lyrics from any English Language inputs. Given a sequence of characters from the data, training a model to predict the next character in the sequence. Longer sequences of text can be generated by calling the model repeatedly.
- **Highlights :**
  - Text Processing : `StringLookup`, `tf.strings.unicode_split`, 
  - Layers : `Embeddings`, `GRU`, 

**[Disaster Tweet Prediction : NLP](https://github.com/ayushs0911/Projects/blob/main/Disaster_tweets_Predictor.ipynb)**<br>
- **Goal of Project :**
  - Constructing a Deep Learning Classification model to predict which Tweets are about real disaster and which one aren't
- **Highlights :**
  - Model 1 : `Feed-Forward neural network`
  - Model 2 : `LSTM` model
  - Model 3 : `GRU` model
  - Model 4 : `Bidirectional-LSTM` model
  - Model 5 : `1D CNN`
  - Model 6 : `TensorFlow Hub Pretrained Feature Extractor`

# Time Forecasting <a name="ts"></a>
**Sales Department Case Study : [Sales Forecasting](https://github.com/ayushs0911/Projects/blob/main/Sales_Forecast_using_Facebook_Prophet.ipynb)**
- **Goal of Project :** 
  - Forecast sales using store, promotion, and competitor data
  - For companies to become competitive and skyrocket their growth, they need to leaverage AI/ML to develop predictive models to forecast sales in future. Predictive models attempt at forceasting future sales based on historical data while taking into account seasonality effects, demand, holidays, promotions, and competition.
- **The Process:**
  - Dataset from Kaggle : `Sales Data`, `Store information Data` | Data Cleaning : Checked Nulls, Dropped not-important columns,
  - `Merged` both dataset on 'Store Dataset'
  - Statistical Anlaysis 
  - Used Facebook `Prophet Alogorithm` for Prediction. 


# Misc <a name="misc"></a>
- [OpenCV](https://github.com/ayushs0911/OpenCV)
- [Google Deep Dream](https://github.com/ayushs0911/DeepDream) : The DeepDream algorithm effective in asking a pre-trained CNN to take a look at the image, identify patterns you recognise and amplify it. It uses representations learned by CNNs to produce these hallucinogenic or 'trippy' images.<br>

