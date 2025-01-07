from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load and preprocess the dataset
file_path = "DmartSalesData.csv"
df = pd.read_csv(file_path)

# Drop rows with missing values in selected columns
df = df.dropna(subset=['Total', 'gross income', 'Rating', 'Quantity'])

# Prepare the data for clustering
X = df[['Total', 'gross income', 'Rating', 'Quantity']].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)

df['Cluster'] = dbscan_labels

# Compute cluster descriptions
cluster_descriptions = df.groupby('Cluster')[['Total', 'gross income', 'Rating', 'Quantity']].mean()

# Flask app setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', cluster=None, description=None, message=None, error=None)

@app.route('/get-cluster-description', methods=['POST'])
def get_cluster_description():
    try:
        # Parse the input data
        input_data = request.form
        
        # Check if any field is missing
        total = input_data.get('Total')
        gross_income = input_data.get('Gross Income')
        rating = input_data.get('Rating')
        quantity = input_data.get('Quantity')

        if not total or not gross_income or not rating or not quantity:
            return render_template('index.html', error="Please fill in all fields.")

        # Convert inputs to float
        data_point = [
            float(total),
            float(gross_income),
            float(rating),
            float(quantity)
        ]

        # Scale the input data
        data_point_scaled = scaler.transform([data_point])

        # Predict the cluster
        cluster_label = dbscan.fit_predict(np.vstack([X_scaled, data_point_scaled]))[-1]

        if cluster_label == -1:
            return render_template('index.html', message="The data point belongs to noise.")

        # Get the cluster description
        cluster_description = cluster_descriptions.loc[cluster_label].to_dict()

        return render_template('index.html', cluster=int(cluster_label), description=cluster_description)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
