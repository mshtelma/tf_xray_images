# Databricks notebook source
# MAGIC %md
# MAGIC # Consumer Pipeline

# COMMAND ----------

# DBTITLE 1,Set up MLflow Experiment
experimentID = '2147458529668738'
model_name = 'nih_xray_model'

# COMMAND ----------

# DBTITLE 1,Model inference at scale using Spark and Pandas UDF
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql import Row
from pyspark.sql.types import ArrayType, DoubleType, StringType, FloatType, LongType
import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.keras

@tf.function(autograph=False)
def parse_image(img):
    return tf.image.per_image_standardization(tf.reshape(tf.io.decode_raw(img, tf.uint8) , (224, 224, 3)))
  
@pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR_ITER)
def predict_batch_udf(image_batch_iter):  
  batch_size = 96
  model = mlflow.keras.load_model('models:/{}/{}'.format(model_name, 'Production'))
  for image_batch in image_batch_iter:
    dataset = tf.data.Dataset.from_tensor_slices(image_batch)
    dataset = dataset.map(parse_image).batch(batch_size)
    preds = model.predict(dataset)
    yield pd.Series(list(preds))

df = spark.read.format("delta").load("/Users/msh/nihxray/nih_xray.delta").sample(0.01)
predictions_df = df.select(col('image'), predict_batch_udf(col("image")).alias("prediction"))
display(predictions_df)
