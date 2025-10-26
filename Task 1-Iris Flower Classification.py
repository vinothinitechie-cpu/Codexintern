import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_and_explore_data():
    """
    Loads the Iris dataset, converts it to a DataFrame, and performs basic exploration.
    
    Returns:
        pd.DataFrame: The Iris dataset as a DataFrame with species names.
    """
    print("🌸 Loading Iris dataset...")
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['species'] = iris.target
    data['species'] = data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("✅ Dataset loaded successfully!")
    print("First 5 Rows:\n", data.head())
    print("\nDataset Info:")
    print(data.info())
    
    # Visual exploration
    print("📊 Generating pairplot for visual analysis...")
    try:
        sns.pairplot(data, hue="species")
        plt.show()
        print("✅ Pairplot displayed!")
    except Exception as e:
        print(f"❌ Error displaying plot: {e}. Ensure matplotlib and seaborn are installed.")
    
    return data

def split_and_scale_data(data, test_size=0.2, random_state=42):
    """
    Splits the data into train/test sets and applies feature scaling.
    
    Args:
        data (pd.DataFrame): The dataset.
        test_size (float): Proportion for test set (default: 0.2).
        random_state (int): Random seed for reproducibility.
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print("✅ Data split into train/test sets!")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Feature scaling applied!")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    """
    Trains a Logistic Regression model.
    
    Args:
        X_train: Scaled training features.
        y_train: Training labels.
    
    Returns:
        LogisticRegression: The trained model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("✅ Model training completed!")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on test data and prints metrics.
    
    Args:
        model: The trained model.
        X_test: Scaled test features.
        y_test: Test labels.
    """
    y_pred = model.predict(X_test)
    print("\n📊 Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def predict_custom_sample(model, scaler):
    """
    Allows interactive prediction on user-provided features.
    
    Args:
        model: The trained model.
        scaler: The fitted scaler.
    """
    print("\n🧪 Test on Custom Sample:")
    try:
        sepal_length = float(input("Enter sepal length (cm): "))
        sepal_width = float(input("Enter sepal width (cm): "))
        petal_length = float(input("Enter petal length (cm): "))
        petal_width = float(input("Enter petal width (cm): "))
        
        sample = [[sepal_length, sepal_width, petal_length, petal_width]]
        sample_scaled = scaler.transform(sample)
        prediction = model.predict(sample_scaled)
        print(f"🔍 Predicted species: {prediction[0]}")
    except ValueError:
        print("❌ Invalid input. Please enter numeric values.")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

# Main execution flow
if __name__ == "__main__":
    print("🚀 Starting Iris Flower Classification Project...")
    
    # Step 1: Load and explore data
    data = load_and_explore_data()
    
    # Step 2: Split and scale data
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(data)
    
    # Step 3: Train model
    model = train_model(X_train, y_train)
    
    # Step 4: Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Step 5: Interactive prediction
    predict_custom_sample(model, scaler)
    
    print("🎉 Project completed! Feel free to run again.")
