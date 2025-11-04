# Databricks notebook source
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
import numpy as np
from PIL import Image
import io

def image_to_features(content):
    try:
        img = Image.open(io.BytesIO(content)).resize((64, 64)).convert('RGB')
        arr = np.array(img).flatten() / 255.0
        return arr.tolist()
    except Exception:
        return [0.0]*(64*64*3)

image_udf = udf(image_to_features, ArrayType(FloatType()))


# COMMAND ----------

display(dbutils.fs.ls("/FileStore/corel_images"))  #Verify image files & structure on DBFS
display(dbutils.fs.ls("/FileStore/corel_images/training_set"))


# COMMAND ----------

df = spark.read.format("binaryFile").option(
    "pathGlobFilter", "*.jpg"
).load("/FileStore/corel_images/training_set/*/*")
display(df)   #ingest data

# COMMAND ----------

sample_df = df.sample(False, 0.1)  # Adjust the reduces dataset size for faster testing.
display(sample_df)


# COMMAND ----------

import mlflow
mlflow.set_experiment("/Shared/CorelImageClassification")


# COMMAND ----------

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "simple_cnn")
    mlflow.log_param("image_size", "128x128")
    
    # Training code goes here
    # e.g. accuracy = train_model(...)
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.85)
    
    # Log the trained model for reproducibility (sample for sklearn, pytorch etc.)
    # For example, using PyTorch:
    # mlflow.pytorch.log_model(model, "model")


# COMMAND ----------

import shutil
from PIL import Image
import numpy as np

def extract_features(row):
    dbfs_path = row['path']
    local_path = "/tmp/" + dbfs_path.split("/")[-1]
    # Copy file from DBFS to local file system
    dbutils.fs.cp(dbfs_path, "file:" + local_path)
    img = Image.open(local_path).resize((64, 64))
    arr = np.array(img).flatten() / 255.0
    return arr


# COMMAND ----------

import io
from PIL import Image
import numpy as np

def extract_features(row):
    img_path = row['path'].replace('dbfs:', '/dbfs')
    with open(img_path, 'rb') as f:
        img = Image.open(f).resize((64, 64))
        arr = np.array(img).flatten() / 255.0
    return arr


# COMMAND ----------

df = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.jpg") \
    .load("/FileStore/corel_images/training_set/*/*")


# COMMAND ----------

from pyspark.sql.functions import regexp_extract #LABEL EXTRACTION,Makes a new column 'label' based on each image’s folder.
label_regex = r"/training_set/([^/]+)/"
df = df.withColumn('label', regexp_extract('path', label_regex, 1))


# COMMAND ----------

df_feats = df.withColumn('features', image_udf(df['content'])) #Converts raw image bytes to numeric ML features in distributed fashion.


# COMMAND ----------

rows = df_feats.select('features', 'label').collect()
#Collect Features/Labels for ML
import numpy as np
X = np.array([row['features'] for row in rows])
y = np.array([row['label'] for row in rows]) #X is features; y is class labels.


# COMMAND ----------

dbutils.fs.mkdirs("dbfs:/FileStore/corel_images")


# COMMAND ----------

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(y)  # y should be your list/array of class labels (e.g., y = np.array([...]) from previous steps)


# COMMAND ----------

import joblib
temp_path = "/tmp/label_encoder.joblib"
joblib.dump(label_encoder, temp_path)


# COMMAND ----------

import mlflow
import mlflow.sklearn
mlflow.sklearn.log_model(label_encoder, "label_encoder")


# COMMAND ----------

import mlflow.sklearn

# Load the label_encoder model from the logged MLflow run
label_encoder_loaded = mlflow.sklearn.load_model("runs:/d4c20dfd7bf849f0b6234e2bf7e22573/label_encoder")


# COMMAND ----------

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)   # should already be defined as your label array


# COMMAND ----------

from sklearn.model_selection import train_test_split #Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# COMMAND ----------

# Run this cell if you get the "already active" error in a notebook
import mlflow
mlflow.end_run()
print("MLflow run ended.")

# COMMAND ----------

import mlflow  #Model Training and Logging with MLflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment("/Shared/CorelImageClassification")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mlflow.log_param("n_estimators", 50)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(
        model,
        "model",
        input_example=np.array([X_train[0]])
    )
    print("Test accuracy:", acc)



# COMMAND ----------

mlflow.register_model("runs:/ec6198cbb5454c5aa177fdfc299982fe/model", "CorelImageClassifier") #Register Model in MLflow Registry
