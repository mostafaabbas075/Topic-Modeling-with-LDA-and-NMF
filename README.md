# Topic Modeling & Text Analysis Project

## 📌 Overview
This project demonstrates **Topic Modeling** using both **LDA (Latent Dirichlet Allocation)** and **NMF (Non-negative Matrix Factorization)**, along with **WordCloud visualizations** for extracted topics.

## 📂 Contents
- Data loading and preprocessing
- Tokenization, lemmatization, and stopword removal
- Building dictionary & corpus for topic modeling
- Training **LDA** and **NMF** models
- Displaying and visualizing topics
- WordCloud generation for each topic

## 📊 Dataset
- **Google Drive (Notebook + Dataset)**: [Drive Link](https://drive.google.com/drive/folders/1GPwNE_hfN991rZyHfwZE07i0-gfXxjfM?usp=sharing)
- **Kaggle Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/gpreda/bbc-news)

## 🚀 Requirements
pip install pandas numpy re spacy gensim pyLDAvis wordcloud matplotlib scikit-learn
python -m spacy download en_core_web_sm

## ⚙️ Usage
# Path to dataset
data_path = '/path/to/your/dataset.csv'

# Create TopicModeling object
tm = TopicModeling(data_path)

# Load and preprocess data
tm.load_data()
tm.preprocess_data()
tm.build_dictionary_corpus()

# Train and display LDA
tm.train_lda(num_topics=5)
tm.display_topics_lda()

# Train and display NMF
tm.train_nmf(num_topics=5)
tm.display_topics_nmf()

# Generate WordClouds
tm.plot_wordcloud()

## 📌 Output
- Printed list of topics from **LDA** and **NMF**
- **Interactive pyLDAvis visualization** for topic exploration
- **WordCloud images** for each topic

## ✍️ Author  
Mostafa Abbas Saleh  
AI Student | NLP Practitioner

## 🙏 Acknowledgment  
Thanks to **Elevvo** for the valuable internship experience and professional training.
