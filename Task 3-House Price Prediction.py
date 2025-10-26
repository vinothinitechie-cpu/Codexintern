import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

def load_and_explore_data():
    """
    Loads the Boston Housing dataset, converts to DataFrame, and explores distributions.
    
    Returns:
        pd.DataFrame: The dataset with features and target (MEDV as price).
    """
    print("🏠 Loading Boston Housing dataset...")
    boston = load_boston()
    data = pd.DataFrame(boston.data, columns=boston.feature_names)
    data['PRICE'] = boston.target  # Target: Median value of owner-occupied homes
    
    print("✅ Dataset loaded successfully!")
    print("First 5 Rows:\n", data.head())
    print("\nDataset Info:")
    print(data.info())
    print("\nBasic Statistics:\n", data.describe())
    
    # Visualize distributions
    print("📊 Generating histograms for feature distributions...")
    try:
        data.hist(bins=30, figsize=(15, 10))
        plt.tight_layout()
        plt.show()
        print("✅ Histograms displayed!")
    except Exception as e:
        print(f"❌ Error displaying plots: {e}. Ensure matplotlib is installed.")
    
    return data

def preprocess_data(data):
    """
    Handles missing data (imputation) and applies normalization.
    
    Args:
        data (pd.DataFrame): The raw dataset.
    
    Returns:
        tuple: (X_scaled, y, scaler) where X_scaled is normalized features, y is target, scaler is the fitted StandardScaler.
    """
    # Check for missing values
    print("🔍 Checking for missing values...")
    if data.isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy='mean')
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        print("✅ Missing values imputed with mean.")
    else:
        print("✅ No missing values found.")
    
    X = data.drop('PRICE', axis=1)  # Features
    y = data['PRICE']               # Target
    
    # Normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Feature normalization applied!")
    
    return X_scaled, y, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits data into train/test sets.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        test_size (float): Proportion for test set.
        random_state (int): Random seed.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print("✅ Data split into train/test sets!")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Trains a Linear Regression model.
    
    Args:
        X_train: Training features.
        y_train: Training target.
    
    Returns:
        LinearRegression: The trained model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Model training completed!")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using regression metrics.
    
    Args:
        model: The trained model.
        X_test: Test features.
        y_test: Test target.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n📊 Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.2f}")
    
    # Optional: Scatter plot of predictions vs actual
    print("📈 Generating scatter plot for predictions vs actual...")
    try:
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.title("Actual vs Predicted House Prices")
        plt.show()
        print("✅ Scatter plot displayed!")
    except Exception as e:
        print(f"❌ Error displaying plot: {e}.")

def predict_custom_sample(model, scaler, feature_names):
    """
    Allows interactive prediction on user-provided features.
    
    Args:
        model: The trained model.
        scaler: The fitted scaler.
        feature_names (list): List of feature names for input prompts.
    """
    print("\n🧪 Test on Custom Sample:")
    try:
        sample = []
        for feature in feature_names:
            value = float(input(f"Enter {feature}: "))
            sample.append(value)
        
        sample_scaled = scaler.transform([sample])
        prediction = model.predict(sample_scaled)
        print(f"🔍 Predicted house price: ${prediction[0]:.2f} (in thousands)")
    except ValueError:
        print("❌ Invalid input. Please enter numeric values.")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

# Main execution flow
if __name__ == "__main__":
    print("🚀 Starting House Price Prediction Project...")
    
    # Step 1: Load and explore data
    data = load_and_explore_data()
    
    # Step 2: Preprocess data
    X, y, scaler = preprocess_data(data)
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 4: Train model
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Step 6: Interactive prediction
    feature_names = list(data.drop('PRICE', axis=1).columns)
    predict_custom_sample(model, scaler, feature_names)
    
    print("🎉 Project completed! Feel free to run again.")
