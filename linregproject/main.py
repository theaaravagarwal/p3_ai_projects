import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import kagglehub
import os

download_path = os.path.join(os.getcwd(), "housedata") #download to housedata dir in current folder
os.makedirs(download_path, exist_ok=True)

# Download latest version
path = kagglehub.dataset_download("shree1992/housedata", path=download_path)

print("Path to dataset files:", path)

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

def gradient_descent(X, y, theta, learning_rate, epochs):
    m = len(y)
    cost_history = []
    theta_history = []
    for i in range(epochs):
        predictions = X.dot(theta)
        gradients = (1/m) * X.T.dot(predictions - y)
        theta = theta - learning_rate * gradients
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        theta_history.append(theta.copy())
        if i % 100 == 0:
            print(f"Epoch {i}: Cost = {cost:.2f}")
    return theta, cost_history, theta_history

def main():
    print("=== HOUSE PRICE PREDICTION MODEL ===")
    print("Loading dataset from " + path + "...")

    df = pd.read_csv(path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total samples: {len(df)}")

    features = ["sqft_living", "bedrooms", "bathrooms", "yr_built", "sqft_lot"]
    X = df[features]
    y = df["price"]
    
    print(f"\nSelected features: {features}")
    print(f"Target variable: price")
    
    X = X.to_numpy()
    y = y.to_numpy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\nFeature statistics (after scaling):")
    for i, feature in enumerate(features):
        print(f"{feature}: mean={X_scaled[:, i].mean():.2f}, std={X_scaled[:, i].std():.2f}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"\nData split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    X_train_with_bias = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test_with_bias = np.c_[np.ones(X_test.shape[0]), X_test]
    
    n_features = X_train_with_bias.shape[1]
    initial_theta = np.random.normal(0, 0.01, n_features)
    
    learning_rate = 0.01
    epochs = 1000
    
    print(f"\n=== INITIAL VALUES ===")
    print(f"Initial parameters (theta): {initial_theta}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {epochs}")
    initial_cost = compute_cost(X_train_with_bias, y_train, initial_theta)
    print(f"Initial cost: {initial_cost:.2f}")
    
    print(f"\n=== TRAINING MODEL ===")
    final_theta, cost_history, theta_history = gradient_descent(
        X_train_with_bias, y_train, initial_theta, learning_rate, epochs
    )
    
    print(f"\n=== FINAL VALUES ===")
    print(f"Final parameters (theta): {final_theta}")
    final_cost = cost_history[-1]
    print(f"Final cost: {final_cost:.2f}")
    print(f"Cost reduction: {initial_cost - final_cost:.2f}")
    
    y_train_pred = X_train_with_bias.dot(final_theta)
    y_test_pred = X_test_with_bias.dot(final_theta)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n=== MODEL PERFORMANCE ===")
    print(f"Training MSE: {train_mse:.2f}")
    print(f"Testing MSE: {test_mse:.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")

    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    ax1.plot(cost_history, color='blue', linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Cost')
    ax1.set_title('Cost Function Over Training Epochs')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(y_test, y_test_pred, alpha=0.6, color='blue')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual Price')
    ax2.set_ylabel('Predicted Price')
    ax2.set_title('Actual vs Predicted House Prices')
    ax2.grid(True, alpha=0.3)
    
    theta_array = np.array(theta_history)
    for i in range(min(3, theta_array.shape[1])):
        param_name = 'Intercept' if i == 0 else features[i-1]
        ax3.plot(theta_array[:, i], label=param_name, linewidth=2)
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Parameter Value')
    ax3.set_title('Parameter Evolution During Training')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    feature_names = ['Intercept'] + features
    ax4.bar(feature_names, final_theta, color=['red'] + ['blue'] * len(features))
    ax4.set_xlabel('Features')
    ax4.set_ylabel('Coefficient Value')
    ax4.set_title('Final Model Coefficients')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linearized_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'linearized_plot.png'")
    print("=== MODEL TRAINING COMPLETE ===")

if __name__ == "__main__":
    main()