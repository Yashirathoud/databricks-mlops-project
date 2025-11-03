# 02_batch_inference.py - Code to run as a Databricks Job for batch inference

import dbutils
import numpy as np
import pandas as pd
from PIL import Image
import io
import mlflow

# Spark and Utilities
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract

# 1. PARAMETER RETRIEVAL (From GitHub Actions)
# The inference pipeline passes the target model URI via base-parameters
try:
    # e.g., models:/databricks_ws.default.corelimageclassifier/1 or @latest
    MODEL_URI = dbutils.widgets.get("model_uri")
except Exception:
    # Fallback for interactive run/testing
    MODEL_URI = "models:/databricks_ws.default.corelimageclassifier@latest" 
    print(f"Failed to retrieve MODEL_URI parameter. Using default: {MODEL_URI}")

# Initialize Spark Session
spark = SparkSession.builder.appName("CorelInferenceJob").getOrCreate()

# Constants
IMAGE_SIZE = 64
FEATURE_SIZE = IMAGE_SIZE * IMAGE_SIZE * 3

# 2. FEATURE EXTRACTION FUNCTION (Local - used after collection)
def extract_features_and_label(path, content):
    """Processes a single binary file record into features and extracts label."""
    try:
        # Image Processing
        img = Image.open(io.BytesIO(content)).resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
        
        # Flatten and normalize
        # NOTE: We do not convert to float64 here; conversion happens in the final NumPy stack
        arr = np.array(img).flatten() / 255.0
        
        # Label extraction logic replicated here for simplicity
        # Assumes format like: /FileStore/corel_images/test_set/label/image.jpg
        label_match = regexp_extract(path, r"/test_set/([^/]+)/", 1)
        label = label_match if label_match else "unknown"

        return {
            "path": path,
            "features": arr.tolist(), # Store as list
            "label": label
        }
    except Exception as e:
        print(f"Skipping bad image {path}: {e}")
        return None

# 3. DATA LOADING and PREPARATION
# Load sample data for inference (adjust path as needed)
df_raw = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.jpg") \
    .load("/FileStore/corel_images/test_set/*/*") # Use a test/unseen path

# Collect data for local processing (necessary for MLflow pyfunc model)
collected_data = df_raw.collect()

# Process data and create a Pandas DataFrame
records = []
for row in collected_data:
    features = extract_features_and_label(row['path'], row['content'])
    if features:
        records.append(features)

if not records:
    print("No valid images found for inference. Exiting.")
    dbutils.notebook.exit("Inference Failed: No data.")

batch_df = pd.DataFrame(records)
print(f"Loaded {len(batch_df)} records for inference.")

# 4. LOAD MODEL
print(f"Loading Model from URI: {MODEL_URI}")
model = mlflow.pyfunc.load_model(MODEL_URI)

# 5. RUN INFERENCE (with the DTYPE FIX)
# Stack the feature lists into a NumPy array
# CRITICAL FIX: Cast the final stacked array to np.float64 to match the model's signature
features_np = np.stack(batch_df['features'].values).astype(np.float64)

# Validation check
if features_np.shape[1] != FEATURE_SIZE:
    raise ValueError(f"Feature shape mismatch. Expected: {FEATURE_SIZE}, Found: {features_np.shape[1]}.")

# Batch prediction
predictions = model.predict(features_np)

# 6. POST-PROCESSING AND OUTPUT
batch_df["prediction_encoded"] = predictions
# NOTE: To get human-readable labels, you would need to load the label_encoder artifact
# and use label_encoder.inverse_transform(predictions). We skip that here for brevity.

print("\n--- Inference Results Sample ---")
print(batch_df[["path", "label", "prediction_encoded"]].head())

# Optionally, save results to a Delta table
# spark.createDataFrame(batch_df).write.mode("overwrite").saveAsTable("databricks_ws.default.corel_inference_results")

dbutils.notebook.exit("Inference Complete and Successful.")