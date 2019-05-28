
# FYP-II
## All material related to Final Year Project II will be uploaded here

# Project Details:
1. Scrap twitter tweets data and use twitter search API to build a dataset.
2. Annotating the data and assigning the class: PI or No PI where PI = purchase intention.
3. Initial survey of data which invloves looking at the type of tweets, checking for imbalance class, building word clouds and different graphical representations to visual the data.
4. Use different pre processing techniques on the data to build a corpus.
5. Apply different machine learing text analytical models on the dataset.
    1. Support Vector Machine
    2. Naive Bayes
    3. Logistic Regression
    4. Decision Tree
    5. Neural Network
6. Output a list of customers who have shown purchase intention towards the product.
7. Develop a website to display the summary of our work and allow users to upload their dataset and train and/or test the pre-developed models for evaluation.

## Set up dependencies for Project Website:
1. You must have python(latest version), django, numpy, pandas, nltk, textblob, sklearn,  installed on your system.
2. To install use folloowing syntax given below on commad prompt of Windows, here package-name corespond to django, numpy, pandas, nltk, textblob, sklearn:
    1. pip3 install package-name 
3. Use the following syntax to import textblob corpora:
    python -m textblob.download_corpora
4. Open Python terminal on windows command promt and install nltk following library:
    1. import nltk
    2. nltk.download('stopwords')
    3. exit()
5. All required dependencies are installed now.

## How to Run Website:
1. Clone and download the complete repository and UnZip it.
2. Using Windows Commmand Prompt(cmd) navigate to folder PurchaseIntention2 in the cloned repository.
3. Then Type "Scripts\activate" on cmd to activate the virtual environment.
4. Then Type "cd djangoPIWebsite" on cmd.
5. Then finally type "python manage.py runserver" to run the server.
5. On browser type "localhost:8000/"
6. The Web Site is running now.
7. To close server press Ctrl + C on cmd to exit the server. To re run type "python manage.py runserver" on cmd.

## How to run simple Models File:
1. change working directory to "CIP\PurchaseIntention2".
2. Then Type "Scripts\activate" on cmd to activate the virtual environment.
3. change working directory to "CIP\PurchaseIntention2\djangoWebsite\pages".
3. open python terminal in cmd by typing "python"
4. Then type "import ModelTest as mt" on terminal.
5  Then type mt.output_to_results("Annotated4.csv","AnnotatedData2.csv", "TF-IDF", "Naive Bayes","90","80","70")
6. Output will show prediction results and accuracy score for model tested.

