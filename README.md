# 🛒 Predicting Consumer Purchase Intention using Twitter Data

> **Final Year Project II (FYP-II)** — All material related to this project is uploaded here.

---

## 📖 Overview

This project leverages **Twitter data** and **machine learning** to predict whether a user has **purchase intention (PI)** towards a product. By analysing tweet content, we classify users as either showing purchase intention or not, and surface actionable insights through an interactive web application.

---

## 🎯 Project Details

1. 🐦 **Data Collection** — Scrape Twitter tweets using the Twitter Search API to build a labelled dataset.
2. 🏷️ **Annotation** — Label each tweet as **PI** (Purchase Intention) or **No PI**.
3. 📊 **Exploratory Data Analysis** — Inspect tweet types, check for class imbalance, generate word clouds and visualisations.
4. 🔧 **Pre-processing** — Apply text cleaning and feature-extraction techniques to build a corpus.
5. 🤖 **Machine Learning Models** — Train and evaluate multiple classifiers:
   - Support Vector Machine (SVM)
   - Naive Bayes
   - Logistic Regression
   - Decision Tree
   - Neural Network
6. 📋 **Output** — Produce a ranked list of customers who have shown purchase intention.
7. 🌐 **Web Application** — A Django-powered website that summarises results and lets users upload their own dataset to train/test the pre-built models.

---

## 🗂️ Repository Structure

| Folder / File | Description |
|---|---|
| `data/` | Annotated CSV datasets used for training and testing |
| `models/` | Python scripts for each ML model and pre-processing pipeline |
| `posters/` | Project poster and standee artwork |
| `SYMPOSIUM/` | Presentation slides, final report, and project completion certificate |
| `PurchaseIntention2.zip` | Complete Django website source code |
| `Model Evaluation - Sheet1.pdf` | Comparative model evaluation results |

---

## ⚙️ Setting Up Dependencies

Make sure you have **Python (latest version)** installed, then install the required packages.

### 📦 Install packages

```bash
pip3 install django numpy pandas nltk textblob scikit-learn
```

### 📥 Download TextBlob corpora

```bash
python -m textblob.download_corpora
```

### 📥 Download NLTK stopwords

Open a Python terminal and run:

```python
import nltk
nltk.download('stopwords')
exit()
```

✅ All required dependencies are now installed.

---

## 🚀 How to Run the Website

1. **Clone** the repository and **unzip** `PurchaseIntention2.zip`.
2. Open **Windows Command Prompt (cmd)** and navigate to the `PurchaseIntention2` folder.
3. Activate the virtual environment:
   ```bash
   Scripts\activate
   ```
4. Navigate to the Django project:
   ```bash
   cd djangoPIWebsite
   ```
5. Start the development server:
   ```bash
   python manage.py runserver
   ```
6. Open your browser and go to **[localhost:8000](http://localhost:8000/)** 🎉
7. To stop the server press **Ctrl + C**. To restart, re-run `python manage.py runserver`.

---

## 🧪 How to Run the Model Scripts Directly

1. Navigate to `CIP\PurchaseIntention2` and activate the virtual environment:
   ```bash
   Scripts\activate
   ```
2. Navigate to the pages directory:
   ```bash
   cd CIP\PurchaseIntention2\djangoWebsite\pages
   ```
3. Open a Python terminal:
   ```bash
   python
   ```
4. Import and run the model test module:
   ```python
   import ModelTest as mt
   mt.output_to_results("Annotated4.csv", "AnnotatedData2.csv", "TF-IDF", "Naive Bayes", "90", "80", "70")
   ```
5. 📈 The output will display prediction results and the accuracy score for the tested model.

---

## 📜 License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.

