# Databricks notebook source
# MAGIC %md 
# MAGIC # Preparation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Libraries

# COMMAND ----------

# MAGIC %pip install petastorm pillow

# COMMAND ----------

# MAGIC %md
# MAGIC # Image processing with Apache Spark and Petastorm

# COMMAND ----------

# MAGIC %md
# MAGIC ## Image Loading

# COMMAND ----------

# DBTITLE 1,Let's read source images with Spark
from pyspark.sql.types import *
from pyspark.sql.functions import *
import PIL.Image
import numpy as np

images_dir = '/mnt/databricks-datasets-private/ML/nih_xray/images/'
raw_image_df = spark.read.format("image").load(images_dir)

display(raw_image_df)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

images = raw_image_df.where("image.nChannels=4").select(col("image.data").alias("image")).take(10)

fig, ax = plt.subplots(1, 2)

ax[0].imshow(np.asarray(images[0]["image"], dtype='uint8').reshape(1024,1024, 4), cmap='gray')
ax[1].imshow(np.asarray(images[9]["image"], dtype='uint8').reshape(1024,1024, 4), cmap='gray')

display()
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Image Conversion

# COMMAND ----------

# DBTITLE 1,We can convert images with pandas_udf
from pyspark.sql.types import StringType, DoubleType, IntegerType, ArrayType
from pyspark.sql.functions import udf
import PIL.Image
import numpy as np

def convert(imm):
    data = np.array(imm.data, dtype='uint8')
    
    if imm.nChannels == 4: 
        image = PIL.Image.frombytes('RGBA', (1024,1024), data.reshape((1024, 1024, 4)))
    else: 
        image = PIL.Image.frombytes('L', (1024,1024), data.reshape((1024, 1024)))
        
    image = image.convert('RGB').resize((224, 224), resample=PIL.Image.LANCZOS)
    
    return np.array(image.getdata(), dtype='uint8').reshape((224, 224, 3)).tobytes()

convert_udf = udf(convert, BinaryType())

to_filename_udf = udf(lambda f: f.split("/")[-1], StringType())

image_df = raw_image_df.select(to_filename_udf("image.origin").alias("origin"), convert_udf("image").alias("image"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Labelling

# COMMAND ----------

# DBTITLE 1,Load and convert labels ...
from pyspark.sql.functions import *

raw_metadata_df = spark.read.\
    option("header", True).option("inferSchema", True).\
    csv("/mnt/databricks-datasets-private/ML/nih_xray/metadata/").\
    select("Image Index", "Finding Labels")

distinct_findings = sorted([
    r["col"] 
    for r in raw_metadata_df.select(explode(split("Finding Labels", "\|"))).distinct().collect() 
    if r["col"] != "No Finding"
])

encode_findings_schema = StructType([
    StructField(f.replace(" ", "_"), IntegerType(), False) 
    for f in distinct_findings
])

def encode_finding(raw_findings):
    findings = raw_findings.split("|")
    return [
        1 if f in findings else 0
        for f in distinct_findings
    ]

encode_finding_udf = udf(encode_finding, encode_findings_schema)

metadata_df = raw_metadata_df.withColumn("encoded_findings", encode_finding_udf("Finding Labels")).select("Image Index", "encoded_findings.*")


# COMMAND ----------

# DBTITLE 1,... and join labels to images DataFrame
df = metadata_df.join(image_df, metadata_df["Image Index"] == image_df["origin"])

combined_df = df.select("origin", "image", array(*distinct_findings).alias("labels"))
display(combined_df)

# COMMAND ----------

combined_df.write.format("delta").save("/Users/msh/nihxray/nih_xray.delta", mode="overwrite")

# COMMAND ----------


