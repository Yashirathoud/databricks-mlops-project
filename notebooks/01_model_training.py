# 01_model_training.py - Code to run as a Databricks Job for training and registration

import dbutils
import json
import numpy as np
import pandas as pd
from PIL import Image
import io

# MLflow and ML Libraries
import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, regexp_extract
from pyspark.sql.types import ArrayType, FloatType
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. PARAMETER RETRIEVAL (From GitHub Actions)
# The training pipeline passes these parameters via base-parameters in train.yml
# Use dbutils.widgets.get() to retrieve them in a job context.
try:
    CATALOG_NAME = dbutils.widgets.get("catalog_name")
    SCHEMA_NAME = dbutils.widgets.get("schema_name")
    FEATURE_SIZE_STR = dbutils.widgets.get("feature_size")
    FEATURE_SIZE = int(FEATURE_SIZE_STR)
    MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.corelimageclassifier"
except Exception as e:
    # Fallback for interactive run/testing
    print(f"Failed to retrieve parameters: {e}. Using defaults.")
    CATALOG_NAME = "databricks_ws"
    SCHEMA_NAME = "default"
    MODEL_NAME = "databricks_ws.default.corelimageclassifier"
    FEATURE_SIZE = 12288 # 64*64*3

# Initialize Spark Session (if running as a non-notebook script)
spark = SparkSession.builder.appName("CorelTrainJob").getOrCreate()

# 2. FEATURE EXTRACTION UDF
# Resize 64x64x3 = 12288 features. Note: UDF output type is float, which is then cast to float64 later.
def image_to_features(content):
    try:
        # Resize, convert to RGB, flatten, and normalize
        img = Image.open(io.BytesIO(content)).resize((64, 64)).convert('RGB')
        arr = np.array(img).flatten() / 255.0
        # CRITICAL FIX: Ensure the array length matches the expected FEATURE_SIZE
        if len(arr) != FEATURE_SIZE:
             raise ValueError(f"Image feature length {len(arr)} does not match expected {FEATURE_SIZE}")
        return arr.tolist()
    except Exception as e:
        print(f"Error processing image: {e}")
        return [0.0] * FEATURE_SIZE # Return zero array on error

# UDF registration using FloatType() array
image_udf = udf(image_to_features, ArrayType(FloatType()))

# 3. DATA LOADING AND PREPROCESSING
# Load data from DBFS
df = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.jpg") \
    .load("/FileStore/corel_images/training_set/*/*")

# Extract Label using regex (assuming folder name is the label)
label_regex = r"/training_set/([^/]+)/"
df = df.withColumn('label', regexp_extract('path', label_regex, 1))

# Apply UDF to extract features
df_feats = df.withColumn('features', image_udf(df['content']))

# Collect Features and Labels
rows = df_feats.select('features', 'label').collect()
X = np.array([row['features'] for row in rows])
y = np.array([row['label'] for row in rows])

# 4. LABEL ENCODING
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Found {len(label_encoder.classes_)} unique classes.")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 5. MLFLOW TRACKING, TRAINING, AND REGISTRATION
mlflow.set_experiment(f"/Shared/CorelImageClassification_{SCHEMA_NAME}")

with mlflow.start_run() as run:
    # 5a. Train Model
    model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # 5b. Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc}")

    # 5c. Log Model and Metrics
    mlflow.log_param("n_estimators", 50)
    mlflow.log_metric("accuracy", acc)
    
    # Log Model (with Input Example)
    mlflow.sklearn.log_model(
        model,
        "model",
        input_example=np.array([X_train[0]], dtype=np.float64) # Ensure input example is float64!
    )
    
    # Log Label Encoder as an artifact for later decoding
    import joblib
    joblib.dump(label_encoder, "/tmp/label_encoder.joblib")
    mlflow.log_artifact("/tmp/label_encoder.joblib")

    # 5d. Register Model to Unity Catalog
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    
    # Register the model using the fully qualified Unity Catalog name
    registered_model_version = mlflow.register_model(model_uri, MODEL_NAME)
    
    print(f"\n--- Model Registration Complete ---")
    print(f"Registered Model Name: {MODEL_NAME}")
    print(f"Registered Version: {registered_model_version.version}")