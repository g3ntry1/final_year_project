import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

# for consideration to the Department of EEE Ajayi Crowther University
def load_energy_data(file_path):
    """
    Load energy data from Excel file
    """
    try:
        # Load the Excel data
        df = pd.read_excel(file_path)
        print(f"Successfully loaded data with {len(df)} records")
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the data, calculate power consumption and perform  cleaning
    """
    # Check for needed columns
    required_columns = ['current', 'voltage', 'property_size_m2', 'power_generated_kw']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None
    
    # Calculate power consumption (P = I * V)
    # Assuming current is in Amperes and voltage is in Volts
    # Power will be in Watts, convert to kWh assuming 24 hours usage
    df['power_consumed_kwh'] = (df['current'] * df['voltage'] / 1000) * 24
    
    # Handle any missing values
    df = df.fillna(0)
    
    return df

def classify_users(df):
    """
    Classify users as consumers (0) or prosumers (1) based on the given criteria
    """
    conditions = [
        # Prosumer conditions: power > 30 kWh, property > 5000m2, generation between 5-10kW
        (df['power_consumed_kwh'] > 30) & 
        (df['property_size_m2'] > 5000) & 
        (df['power_generated_kw'] >= 5) & 
        (df['power_generated_kw'] <= 10)
    ]
    choices = [1]  # 1 for prosumer
    
    # Apply conditions
    df['user_class'] = np.select(conditions, choices, default=0)
    
    # Map numeric classes to labels
    df['user_type'] = df['user_class'].map({0: 'Consumer', 1: 'Prosumer'})
    
    return df

def build_decision_tree(df):
    """
    Build a decision tree classifier based on the data
    """
    # Features for classification
    X = df[['power_consumed_kwh', 'property_size_m2', 'power_generated_kw']]
    y = df['user_class']
    
    # Create decision tree classifier
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(X, y)
    
    # Feature names and class names for visualization
    feature_names = ['Power Consumed (kWh)', 'Property Size (m²)', 'Power Generated (kW)']
    class_names = ['Consumer', 'Prosumer']
    
    return clf, feature_names, class_names

def visualize_tree(clf, feature_names, class_names):
    """
    Visualize the decision tree
    """
    plt.figure(figsize=(15, 10))
    tree.plot_tree(clf, 
                   feature_names=feature_names,
                   class_names=class_names,
                   filled=True,
                   rounded=True)
    plt.savefig('energy_classification_tree.png')
    print("Decision tree visualization saved as 'energy_classification_tree.png'")

def main():
    file_path = input("Enter the path to your Excel file: ")
    
    # Load data
    df = load_energy_data(file_path)
    if df is None:
        return
    
    # Preprocess data
    processed_df = preprocess_data(df)
    if processed_df is None:
        return
    
    # Classify users
    classified_df = classify_users(processed_df)
    
    # Count of consumers and prosumers
    class_counts = classified_df['user_type'].value_counts()
    print("\nClassification Results:")
    print(f"Consumers: {class_counts.get('Consumer', 0)}")
    print(f"Prosumers: {class_counts.get('Prosumer', 0)}")
    
    # Build and visualize decision tree
    clf, feature_names, class_names = build_decision_tree(classified_df)
    visualize_tree(clf, feature_names, class_names)
    
    # Save results to Excel
    output_path = "energy_classification_results.xlsx"
    classified_df.to_excel(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Show a sample of results
    print("\nSample of classification results:")
    print(classified_df[['power_consumed_kwh', 'property_size_m2', 
                         'power_generated_kw', 'user_type']].head(10))

if __name__ == "__main__":
    main()
