# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Pipeline that evaluates model and deploys it to MLflow Model Registry

# COMMAND ----------

# DBTITLE 1,Import packages and set up MLflow experiment
import os

import pandas as pd
import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K

from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset

import matplotlib.pyplot as plt
import seaborn as sns

import os
import time
import mlflow
import mlflow.sklearn
from  mlflow.tracking import MlflowClient

experimentID = '2147458529668738'
model_name = 'nih_xray_model'
num_classes = 14

# COMMAND ----------

def random_contrast_brightness(image):
    return tf.image.random_contrast(tf.image.random_brightness(image, max_delta=0.6), lower=0.1, upper=1.9)  

def basic_transform(image):
    image = tf.reshape(tf.io.decode_raw(image, tf.uint8) , (-1, 224,224, 3))
    return image

def horizontal_flip(image):
    return tf.image.random_flip_left_right(image)
  
def vertical_flip(image):
    return tf.image.random_flip_up_down(image)
  
def post_process(image):
    return  tf.image.per_image_standardization(image)

# COMMAND ----------

from petastorm.spark import SparkDatasetConverter, make_spark_converter

spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///dbfs/ml/petastormcache")
data = spark.read.format("delta").load("/Users/msh/nihxray/nih_xray.delta")
_, testDf = data.select("image", "labels").randomSplit([0.9, 0.01], seed = 42)

test_rows = testDf.count()
print(test_rows)

converter_test = make_spark_converter(testDf)

# COMMAND ----------

# DBTITLE 1,Prepare evaluation dataset
with  converter_test.make_tf_dataset() as test_dataset:

  test_dataset = (test_dataset
      .unbatch()
      .batch(test_rows)
      .map(lambda x: (
          basic_transform(x.image),  
          tf.reshape(tf.cast(x.labels, dtype=tf.uint8), (-1, 14))
      ))
  )

  x_test, y_test = next(iter(test_dataset))

# COMMAND ----------

# DBTITLE 1,We can access MLflow experiments using native Spark
df = spark.read.format("mlflow-experiment").load(experimentID)
display(df)

# COMMAND ----------

# DBTITLE 1,Function that can calculate evaluation metric for our model
from sklearn.metrics import roc_auc_score
def evaluate_model(run_id, X, Y):
    model = mlflow.keras.load_model('runs:/{}/model'.format(run_id))
    probs = model.predict(X, batch_size=192, verbose=0)
    test_aucs = [roc_auc_score(Y[:,i], probs[:,i]) for i in range(num_classes)]
    auc_mean = np.mean(np.array(test_aucs))
    return auc_mean

# COMMAND ----------

# DBTITLE 1,Evaluation logic
from mlflow.exceptions import RestException

def evaluate_all_candidate_models():
    mlflow_client = MlflowClient()

    cand_run_ids = get_candidate_models()
    best_cand_metric, best_cand_run_id = get_best_model(cand_run_ids, x_test, y_test)
    print('Best ROC AUC (candidate models): ', best_cand_metric)

    try:
        versions = mlflow_client.get_latest_versions(model_name, stages=['Production'])
        prod_run_ids = [v.run_id for v in versions]
        best_prod_metric, best_prod_run_id = get_best_model(prod_run_ids, x_test, y_test)
    except RestException:
        best_prod_metric = -1
    print('ROC AUC (production models): ', best_prod_metric)

    if best_cand_metric >= best_prod_metric:
        # deploy new model
        model_version = mlflow.register_model("runs:/" + best_cand_run_id + "/model", model_name)
        time.sleep(15)
        mlflow_client.transition_model_version_stage(name=model_name, version=model_version.version,
                                                     stage="Production")
        print('Deployed version: ', model_version.version)
    # remove candidate tags
    #for run_id in cand_run_ids:
    #    mlflow_client.set_tag(run_id, 'candidate', 'false')

def get_best_model(run_ids, X, Y):
    best_metric = -1
    best_run_id = None
    for run_id in run_ids:
        metric = evaluate_model(run_id, X, Y)
        print('Evaluated model with metric: ', metric)
        if metric > best_metric:
            best_metric = metric
            best_run_id = run_id
    return best_metric, best_run_id

def get_candidate_models():
    spark_df = spark.read.format("mlflow-experiment").load(experimentID)
    pdf = spark_df.where("tags.candidate='true'").select("run_id").toPandas()
    return pdf['run_id'].values

evaluate_all_candidate_models()

# COMMAND ----------


