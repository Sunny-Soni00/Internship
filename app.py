from flask import Flask, request, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyclustering.cluster.kmedoids import kmedoids
import gower
import os

# Initialize Flask app
app = Flask(__name__)

# Directories for saving files
UPLOAD_DIR = "./uploads"
PLOT_DIR = "./plots"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def homepage():
    """Render homepage and handle file upload and clustering."""
    if request.method == "POST":
        try:
            # Get uploaded file
            file = request.files["file"]
            features = request.form["features"]
            if not file or not features:
                return "Please upload a file and specify features."

            # Save the file
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            file.save(file_path)

            # Load the dataset
            df = pd.read_csv(file_path)

            # Extract selected features
            selected_features = [feature.strip() for feature in features.split(",")]
            if not all(feature in df.columns for feature in selected_features):
                return "One or more selected features are not in the dataset."

            # Compute Gower's distance
            data = df[selected_features]
            gower_matrix = gower.gower_matrix(data)

            # Perform K-Medoids clustering with k=5
            initial_medoids = list(range(5))  # First 5 points as initial medoids
            kmedoids_instance = kmedoids(gower_matrix, initial_medoids, data_type="distance_matrix")
            kmedoids_instance.process()
            clusters_result = kmedoids_instance.get_clusters()

            # Assign cluster labels to data
            cluster_labels = np.zeros(len(df))
            for idx, cluster in enumerate(clusters_result):
                cluster_labels[cluster] = idx
            df['Cluster'] = cluster_labels

            # Plot the clusters (if two features are selected)
            if len(selected_features) == 2:
                plot_path = os.path.join(PLOT_DIR, "cluster_plot.png")
                plt.figure(figsize=(8, 6))
                for cluster_idx, cluster_points in enumerate(clusters_result):
                    cluster_data = data.iloc[cluster_points]
                    plt.scatter(cluster_data[selected_features[0]], cluster_data[selected_features[1]], label=f"Cluster {cluster_idx}")
                plt.title("K-Medoids Clustering")
                plt.xlabel(selected_features[0])
                plt.ylabel(selected_features[1])
                plt.legend()
                plt.savefig(plot_path)
                plt.close()
                return send_file(plot_path, mimetype="image/png", as_attachment=True)

            # Save clustered data if more than 2 features
            clustered_file = os.path.join(PLOT_DIR, "clustered_data.csv")
            df.to_csv(clustered_file, index=False)
            return send_file(clustered_file, mimetype="text/csv", as_attachment=True)

        except Exception as e:
            return f"An error occurred: {str(e)}"

    # Render HTML form
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>K-Medoids Clustering</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                margin: 0;
                padding: 0;
            }
            .container {
                width: 80%;
                margin: 0 auto;
                background: #fff;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                margin-top: 50px;
            }
            h1 {
                text-align: center;
                color: #333;
            }
            form {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            label {
                font-weight: bold;
            }
            input[type="file"],
            input[type="text"],
            button {
                padding: 10px;
                font-size: 16px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            button {
                background-color: #28a745;
                color: white;
                cursor: pointer;
            }
            button:hover {
                background-color: #218838;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>K-Medoids Clustering with Gower's Distance</h1>
            <form method="post" enctype="multipart/form-data">
                <label for="file">Upload CSV File:</label>
                <input type="file" name="file" accept=".csv" required><br><br>
                
                <label for="features">Enter Feature Columns (comma-separated):</label>
                <input type="text" name="features" placeholder="e.g., feature1,feature2" required><br><br>
                
                <button type="submit">Cluster</button>
            </form>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)