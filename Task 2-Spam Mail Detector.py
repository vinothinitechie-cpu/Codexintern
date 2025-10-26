import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
import string

def load_and_prepare_data(filepath="spam.csv"):
    """
    Loads the spam dataset, renames columns, and displays a sample.
    
    Args:
        filepath (str): Path to the CSV file (default: 'spam.csv').
    
    Returns:
        pd.DataFrame: The loaded dataframe with 'label' and 'message' columns.
    """
    try:
        df = pd.read_csv(filepath, encoding='latin-1')[['v1', 'v2']]
        df.columns = ['label', 'message']
        print("✅ Dataset loaded successfully!")
        print("Sample Data:\n", df.head())
        return df
    except FileNotFoundError:
        print("❌ Error: File not found. Please ensure 'spam.csv' is in the current directory.")
        return None

def preprocess_text(df):
    """
    Cleans the text data: converts to lowercase, removes punctuation and stopwords.
    
    Args:
        df (pd.DataFrame): Dataframe with a 'message' column.
    
    Returns:
        pd.DataFrame: Dataframe with an added 'cleaned' column.
    """
    print("🔄 Downloading NLTK stopwords (if not already done)...")
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        text = text.lower()  # Convert to lowercase
        text = "".join([ch for ch in text if ch not in string.punctuation])  # Remove punctuation
        words = [word for word in text.split() if word not in stop_words]  # Remove stopwords
        return " ".join(words)
    
    df['cleaned'] = df['message'].apply(clean_text)
    print("✅ Text preprocessing completed!")
    return df

def vectorize_text(df):
    """
    Converts cleaned text to TF-IDF vectors.
    
    Args:
        df (pd.DataFrame): Dataframe with 'cleaned' column.
    
    Returns:
        tuple: (X, y, vectorizer) where X is the feature matrix, y is labels, and vectorizer is the fitted TF-IDF object.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned'])
    y = df['label']
    print("✅ Text vectorization (TF-IDF) completed!")
    return X, y, vectorizer

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Trains a Naive Bayes model and evaluates it.
    
    Args:
        X_train, X_test: Training and test feature matrices.
        y_train, y_test: Training and test labels.
    
    Returns:
        MultinomialNB: The trained model.
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print("✅ Model training completed!")
    
    y_pred = model.predict(X_test)
    print("\n📊 Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model

def test_custom_message(vectorizer, model):
    """
    Tests the model on a custom user-input message.
    
    Args:
        vectorizer: The fitted TF-IDF vectorizer.
        model: The trained model.
    """
    sample_msg = input("💬 Enter a message to test for spam (or press Enter for default): ").strip()
    if not sample_msg:
        sample_msg = "Congratulations! You've won a free iPhone. Click to claim now!"
        print(f"Using default sample: '{sample_msg}'")
    
    sample_msg_vec = vectorizer.transform([sample_msg])
    prediction = model.predict(sample_msg_vec)
    print(f"🔍 Prediction for your message: {prediction[0]}")

# Main execution flow
if __name__ == "__main__":
    print("🚀 Starting Spam Mail Detection Project...")
    
    # Step 1: Load and prepare data
    df = load_and_prepare_data()
    if df is None:
        exit()
    
    # Step 2: Preprocess text
    df = preprocess_text(df)
    
    # Step 3: Vectorize text
    X, y, vectorizer = vectorize_text(df)
    
    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("✅ Train-test split completed!")
    
    # Step 5: Train and evaluate model
    model = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    # Step 6: Test on custom input
    test_custom_message(vectorizer, model)
    
    print("🎉 Project completed! Feel free to run again.")
