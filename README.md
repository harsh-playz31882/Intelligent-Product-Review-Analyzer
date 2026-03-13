# Intelligent Product Review Analyzer

A machine learning system designed to analyze customer reviews and classify sentiment using Natural Language Processing techniques.

## 🎯 Project Overview

This project analyzes Amazon product reviews to classify them as positive, negative, or neutral using multiple machine learning approaches. The system processes raw text data, extracts meaningful features, and trains models to predict sentiment.

## 📊 Dataset

- **Source:** Amazon Fine Food Reviews
- **Size:** 568,454 reviews
- **Features:** Text, Summary, Score (1-5 ratings)
- **Balanced Sample:** 10,000 reviews (3,334 per sentiment class)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/Intelligent-Product-Review-Analyzer.git
cd Intelligent-Product-Review-Analyzer
```

2. **Create virtual environment:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Pipeline

The project is organized into sequential steps. Run them in order:

#### Step 1: Data Exploration
```bash
python src/01_data_exploration.py
```
- Analyzes the dataset structure
- Creates balanced 10k sample
- Generates initial visualizations

#### Step 2: Data Preprocessing
```bash
python src/02_data_preprocessing.py
```
- Cleans and preprocesses text data
- Removes stopwords and applies lemmatization
- Splits data into train/test sets

#### Step 3: Feature Extraction
```bash
python src/03_feature_extraction.py
```
- Extracts TF-IDF features
- Creates Word2Vec embeddings
- Generates text statistics features

#### Step 4: Model Training
```bash
python src/04_model_training.py
```
- Trains multiple ML models
- Evaluates performance
- Generates comparison charts

#### Run Complete Pipeline
```bash
python main.py --step all
```

Or run individual steps:
```bash
python main.py --step 1  # Data exploration
python main.py --step 2  # Preprocessing
python main.py --step 3  # Feature extraction
python main.py --step 4  # Model training
```

## 📁 Project Structure

```
Intelligent-Product-Review-Analyzer/
├── src/                          # Source code
│   ├── 01_data_exploration.py    # Dataset analysis
│   ├── 02_data_preprocessing.py  # Text cleaning
│   ├── 03_feature_extraction.py # Feature engineering
│   └── 04_model_training.py     # ML models
├── data/                         # Dataset storage (empty in repo)
├── models/                       # Trained models (empty in repo)
├── results/                      # Visualizations (empty in repo)
├── main.py                       # Main entry point
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Technical Details

### Data Preprocessing
- **Text Cleaning:** HTML tags, URLs, punctuation removal
- **Tokenization:** Word tokenization with NLTK
- **Stopword Removal:** English stopwords
- **Lemmatization:** WordNet lemmatizer

### Feature Extraction
- **TF-IDF:** 5,000 features, unigrams and bigrams
- **Word2Vec:** 100-dimensional vectors, skip-gram model
- **Text Statistics:** Character count, word count, punctuation

### Machine Learning Models
- **Logistic Regression:** Fast baseline model
- **Naive Bayes:** Probabilistic classifier
- **Random Forest:** Ensemble method
- **SVM:** Support vector machine
- **Gradient Boosting:** Advanced ensemble
- **Neural Network:** Multi-layer perceptron

## 📈 Results

### Performance Metrics (10k dataset)
- **Logistic Regression:** ~85% accuracy
- **Random Forest:** ~87% accuracy
- **Naive Bayes:** ~80% accuracy

### Key Findings
- TF-IDF features perform best for sentiment analysis
- Word2Vec captures semantic relationships
- Balanced dataset crucial for fair evaluation

## 🛠️ Dependencies

```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
nltk==3.8.1
wordcloud==1.9.2
textblob==0.17.1
plotly==5.15.0
stopwords==1.0.1
gensim==4.3.2
```

## 🔍 File Organization

**Important Note:** Large datasets and generated models are stored in `.git/` folder to keep the main repository clean and GitHub-friendly.

- **`.git/data/`**: All CSV files and datasets
- **`.git/models/`**: Trained models and extracted features
- **`.git/results/`**: Visualizations and analysis outputs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Notes for Teammates

### First Time Setup
1. After cloning, the `.git/` folder will contain all necessary data
2. Run steps in sequence (1→2→3→4) to regenerate any missing files
3. All paths are configured to work with the `.git/` folder structure

### Troubleshooting
- **NLTK Data Error:** First run will automatically download required NLTK packages
- **File Not Found:** Ensure you're running scripts from the project root directory
- **Memory Issues:** Use smaller dataset sizes for initial testing

### Customization
- **Dataset Size:** Modify sample size in `01_data_exploration.py`
- **Feature Parameters:** Adjust TF-IDF/Word2Vec settings in `03_feature_extraction.py`
- **Model Selection:** Enable/disable models in `04_model_training.py`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Amazon Fine Food Reviews dataset
- NLTK for natural language processing
- Scikit-learn for machine learning algorithms
- Gensim for Word2Vec implementation

---

**For questions or support, please open an issue in the repository.**
