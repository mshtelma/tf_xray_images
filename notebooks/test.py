# Databricks notebook source
dbfs:/databricks/mlflow-tracking/2147458529668738/07784087cc9c40988d1057844c6a74cb/artifacts/tensorboard_logs

# COMMAND ----------

# MAGIC %load_ext tensorboard

# COMMAND ----------

# MAGIC %tensorboard --logdir /dbfs/databricks/mlflow-tracking/2147458529668738/07784087cc9c40988d1057844c6a74cb/artifacts/tensorboard_logs/train

# COMMAND ----------


