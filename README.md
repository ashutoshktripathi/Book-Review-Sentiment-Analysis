# SENTIMET ANALYSIS ON BOOK REVIEW

Presenting a SENTIMET ANALYSIS Engine to predict positive, neagative and neutral semtiment on BOOK REVIEW dataset.

## SENTIMENT ANALYSIS

Sentiment Analysis or opinion mining is a classification task in Natural Language Processing which predicts the sentiment based on text data. This is of huge interest to businesses and other organisations who wants to understand the public semtiment to their product and services. This is especially useful in present time where huge a of public review and opinions are being genrated on large valume. Its not practical for even a dedicated team to deal with this huge amount of data. As a result researchig and improving Sentimen Engine has become very critical.

## THIS PROJECT

This project is based on deep learning models 

Two seperate models have been developed depending on the length of the review data. 

These are as follows:
- **LTE250** :  This is built for those text data (review data) where total number of words are less than equal to 250. This is built using pre-trained XL-NET with fine tuning. This was built using Transformers library from 
This [link](https://medium.com/swlh/using-xlnet-for-sentiment-classification-cfa948e65e85) helped me alot.
- **GT250** : This is build to deal with larger text with total number of words exceediing 250. This is built combination of Bi-LSTM , Attention and using sentence embdedding from Universal Sentence Encoder. The novel idea being using sentences in a paragraph rather that words as a time series input to the LSTM. - 

## REPOSITORY

Here I will give some informaion on the directories and files and what each directories contain and so on.

- **Exploratory Data Analyis** : This folder contains a single jupyter notebook file named EDA. This has some bsaic analyis like checking any missing values, cheking for category distribution etc.
- **datasets** : This folder contains 3 files.
    - train_data.csv : this is training data which is 80% of all the labelled data from the beginning. All model training is done on this data
    - holdout_data.csv : This is like 20% of the labelled data which was random split (with stratiy) before we performed any modelling and was only used in the end to confirm the model performance.
    - test_data.csv : This data is data with no labels. This is for submission.

- **saved_models** : This folder contains trained models and pre-trained weight. It has 3 main items.
    - xlnet.bin : this is a saved model from **LTE250**. 
    - use5 : This is pre-trained Universal Sentence Encoder Large 5. This is used to get sentence embeddings.
    - tfmodel : TF2 saved model from **GT250**.

- **TrainingNotebooks** : This has the jupyter notebook which was used during experimenation and training the model. It has two notebooks named after the two models LTE250 and GT250. Each notebook has full joureny from data preperation , training and validation. Also the performance of holdout_data is present in this notebook at the end.

- **test** : This folder contains the inference scrips or scripts which can be used to generate predictions based on the saved weights.
It has 3 scripts.
    - gte250classifier.py : this script accepts list of text and list of summary and returns their corresponding sentiments. This will  especially work best with texts with word count greater than 250. 
    - lte250classifer.py : this will expect input as list of text and resturn the sentiments. Especially suited for texts with token lenght less than 250.
    - findsentiment.py : this is a wrapper which combines the prediction scripts of both the models and later combines the result to show the final result. 
    
    In addition to that there is one file names test_data.csv which is the label-less data on which these models predict the sentiment f for evaluaiton.
    There is a file named results.csv which contains the predicted sentiments on above mentioned file.


- **train** : This folder has scripts which anyone can easily use to train their model. THe model is end to end , so one just has to pass the raw train and holdout sets and the model will train, save the model and generate a metric report on performace evalution. More details on how to perform the training is described later. 

- docker flask app : this is a ready to use application for prediction sentiment. More information on how to use it decribed later.




## TRAINING YOURSELF
This section will describe how to use the training scripts to train the model.

**Few Highlights : **
1. **End to End** : Just pass in the train and holdout data. THe model will pre-process, valdate , tokenize and train the model. Also it will save the best model from time and save it in pre defined path. It will automatically do an evaluation on holdout set and generate a classification report and save it to the pre defined folder.
2. **Flexible** : The pipeline is very flexible and supports a lots of option like choosing the paths, model parameter like depths of layers, width of layers, learning parameter, loss function etc.

**HOW TO RUN**

1. Use the train.py file present in the training data.
2. Prepare your train and holdout data into a csv format.
3. This script will automatially segment the texts and use **LTE250** classifer if text is less than 250 words lenght otherwise it will use **GT250**.
4. One will need to provide minimum 6 parameters.
    - train_file_path : path of the train file
    - test_file_path  : path of test file
    - gt_use_path : path or a link to a Universal Sentence Encoder pre-embeddings
    - gt_save_model_path : path to store the saved model for GT250 model.
    - lte_save_model_path : path to store the LTE250 model.
    - lte_save_model_file : file name for save model.
5. Simplest way to run is : 
    
    In terminal type

    `python train.py --train_file_path <somepath> --test_file_path <somepath> --gt_use_path <somepath> --gt_save_model_path <somepath> --lte_save_model_path <somepath> --lte_save_model_file <somename>`

    For full details of params use `python train.py --help`


## DOCKER APPLICATION

One can directly use this sentiment engine without the need to install all the libraries and dependancy. 
Docker has revolutioned the way we can distribute the solutions from person to person or from one system to other. 

#### With the view to make tihs tool easily used by everyone even who dont have programming backgroud, I put the inference solution as a dockerised app.

**This application procvides two modes of run**: 
    1. Sentimet as a Rest Service : The application is running as a web server which accepts WEB API requests and returns with sentiment. Flask becuase of its lightweight and flexible nature has been used as backend service. 
    2. BULK classification : In this mode one can put their csv files in the rquired folder and the application will run and generate the output.


**HOW TO USE THE DOCKER APPLCATION ** : 

- First requirement is to have a DOCKER DESKTOP (if using PC) or DOCKER running on servers.
If one has a KUBERNETES cluster , this app can be easily put there as well.

- Copy the docker flask app folder to some location. 
In a terminal or similar application, naviagte to the folder you downloaded. Make sure the docker-compose.yml file is present.
Now run the following command. `docker-compose up -d` 

It will take some time to download and istall all the dependancy. One has to this only one and later the image is stored locally and everytime it runs from Image instead of building the dependancy. 

After it load all the dependancy RUN `docker ps`. You should see the this contaniner running.

- **Using as an API service.**
    TO run as an API make sure to look in docker-compse file for this line ` command:  /bin/sh -c "python main.py"` The main.py file runs the Rest API mode wheras main2.py run in bulk code.
    To check if its working us API testing tool like PostMan or just open python shell and type:

    import requests

    import json

    headers = {"Content-type": "application/json"}`

    
    data = {
        'text' :    ["great thriller book","worst book I have read"],
        'summary' : ["graet", "very bad"]
       }`
    
    res = requests.post('http://[::1]:8000/', headers = headers, data=json.dumps(data))
    `

    Will the following using res.json()

    `{'data': ['positive', 'negative'], 'status': True}`



- **Running on BULK MODE**

First put your file in the datainput folder same directory as docker-comose file
Now run the docker-compose file with following changes `command:  /bin/sh -c "python main2.py"`

After some time depending on size of data and coputatuion power the results will be generated in dataoutput folder same location as datainput.


## TEST RESULT OF TWO models

### LTE250

![alt text](https://git.toptal.com/screening/Ashutosh-Tripathi/-/raw/master/LTE250ClassificationReport.JPG)



### GT250
![alt text](https://git.toptal.com/screening/Ashutosh-Tripathi/-/raw/master/GT25classificationReport.JPG)





 



