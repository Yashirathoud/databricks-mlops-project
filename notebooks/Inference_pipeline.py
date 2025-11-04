# Databricks notebook source
# Check if Spark is available and cluster is attached
try:
    spark_version = spark.version
    print(f"Spark is available. Cluster version: {spark_version}")
except Exception as e:
    raise RuntimeError("Spark is not available or the cluster is detached. Please attach this notebook to a running cluster and try again.")

import mlflow
from PIL import Image
import numpy as np
import pandas as pd
from io import BytesIO

# --- HELP SECTION: List available models in the workspace registry ---
client = mlflow.tracking.MlflowClient()
print("\nAvailable models in the workspace model registry:")
for rm in client.search_registered_models():
    print(f"- {rm.name}")

print("""
INSTRUCTIONS:
1. Copy the correct model name from the list above (e.g. coreimageclassifier).
2. In the Models UI, find the alias (e.g. 'champion') or version number (e.g. '1') for your model.
3. Set the MODEL_URI variable below:
   - By alias:   MODEL_URI = "models:/<model_name>@<alias>"
   - By version: MODEL_URI = "models:/<model_name>/<version>"
4. Ensure that the model URI is correctly formatted and points to an existing model.
5. If you do not see any models listed, you may not have any models registered, or you may not have permission to view them.
""")

# --- INFERENCE SECTION (update the MODEL_URI below after running the help section above) ---
# Example: MODEL_URI = "models:/coreimageclassifier@champion"
MODEL_URI = "models:/databricks_ws.default.corelimageclassifier/1"  # <-- UPDATE THIS!

if '<' in MODEL_URI or '>' in MODEL_URI:
    raise ValueError("You must set MODEL_URI to a real model path, e.g. 'models:/coreimageclassifier@champion'. See instructions above.")
model = mlflow.pyfunc.load_model(MODEL_URI)

# Load images as Spark DataFrame (binaryFile method)
df = spark.read.format("binaryFile").load("/FileStore/tables/flowers/*.jpg")

# Feature extraction and DataFrame preparation
records = []
for row in df.collect():
    # 1. Open image (PIL Image object) and convert to Grayscale
    img = Image.open(BytesIO(row.content)).convert('L') 
    
    # 2. Add RESIZING step here
    # The target size for a 12288 feature vector (grayscale) is 128x96
    TARGET_SIZE = (128, 96) 
    img_resized = img.resize(TARGET_SIZE)

    # 3. Flatten the resized image
    img_arr = np.array(img_resized).flatten().astype(np.float64).tolist()
    
    records.append({
        "path": row.path,
        # ... other columns
        "features": img_arr
    })

batch_df = pd.DataFrame(records)
batch_df["features"] = batch_df["features"].apply(lambda arr: np.array(arr, dtype=np.float32).tolist())

# --- FIX: Ensure correct shape and dtype for model input ---
# The model expects input shape (-1, 12288) and dtype float32
features_np = np.stack(batch_df['features'].values).astype(np.float64)
expected_shape = 12288
if features_np.shape[1] != expected_shape:
    raise ValueError(f"Model expects input shape (-1, {expected_shape}), but got {features_np.shape}. "
                     f"Check your image preprocessing. Each image should flatten to {expected_shape} features.")

# Batch inference
predictions = model.predict(features_np)
batch_df["prediction"] = predictions.tolist()
print(batch_df[["path", "prediction"]])

# COMMAND ----------

