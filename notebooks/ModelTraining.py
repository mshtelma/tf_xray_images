# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # Configuration

# COMMAND ----------

#prefix = "/Users/msh/nihxray/nih_xray.delta"
#image_path = f"{prefix}/prepared_images"

data = spark.read.format("delta").load("/Users/msh/nihxray/nih_xray.delta")

display(data.limit(1))

# COMMAND ----------

# MAGIC  %md
# MAGIC  # Simple Training

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test / Train Split with Petastorm

# COMMAND ----------

from petastorm.spark import SparkDatasetConverter, make_spark_converter

spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///dbfs/ml/petastormcache")

trainDf, testDf = data.select("image", "labels").randomSplit([0.9, 0.1], seed = 42)

train_rows = trainDf.count()
test_rows = testDf.count()

converter_train = make_spark_converter(trainDf)
converter_test = make_spark_converter(testDf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model using DenseNet architecture

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Concatenate, Activation, Input, Dense, Dropout, Flatten, Lambda, Conv2D, MaxPooling2D, BatchNormalization,Flatten
from tensorflow.keras.optimizers import Nadam, SGD, Adam
from tensorflow.keras.regularizers import l2

num_classes = 14

params = {
    'dropout1': 0.15,
    'dropout2': 0.05,
    'fc1': 256,
    'fc2': 64,
    'fc3': 24,
    'l2': 0.0001
}

def get_model(params):
    from tensorflow.keras.models import Sequential,Model
    from tensorflow.keras.layers import Concatenate, Activation, Input, Dense, Dropout, Flatten, Lambda, Conv2D, MaxPooling2D, BatchNormalization,Flatten
    from tensorflow.keras.optimizers import Nadam, SGD, Adam
    from tensorflow.keras.regularizers import l2

    model = Sequential()

    from tensorflow.keras.applications.densenet import DenseNet121
    dense_net = DenseNet121(include_top=False, weights=None,
                  input_tensor=None, input_shape=(224,224,3),
                  pooling='max', classes=0)

    model.add(dense_net)

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout1']))
    model.add(Dense(params['fc1'],  kernel_regularizer=l2(params['l2'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(params['dropout2']))
    model.add(Dense(params['fc2'],  kernel_regularizer=l2(params['l2'])))  #activation='relu',
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(params['fc3'],  kernel_regularizer=l2(params['l2'])))  #activation='relu',
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='sigmoid', kernel_regularizer=l2(params['l2'])))

    return model

# COMMAND ----------

def basic_transform(image):
    return tf.reshape(tf.io.decode_raw(image, tf.uint8) , (-1, 224, 224, 3))

def random_contrast_brightness(image):
    return tf.image.random_contrast(tf.image.random_brightness(image, max_delta=0.6), lower=0.1, upper=1.9)  

# COMMAND ----------

from petastorm import make_batch_reader, make_reader
from petastorm.tf_utils import make_petastorm_dataset

BATCH_SIZE = 32

with converter_train.make_tf_dataset() as dataset:
    dataset = (dataset
        .unbatch().batch(BATCH_SIZE)
        .map(lambda x: (
            basic_transform(x.image), 
            tf.reshape(tf.cast(x.labels, dtype=tf.uint8), (-1, 14)))
        )
    )
    
    model = get_model(params)
    optimizer = Adam()
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
    )
    
    model.fit(dataset, steps_per_epoch=10, epochs=1)

# COMMAND ----------

dbutils.notebook.exit("Done")

# COMMAND ----------

# MAGIC %md # Model tracking with MLFlow

# COMMAND ----------

# MAGIC %md ## Training and testing

# COMMAND ----------

import time
import mlflow

mlflow.tensorflow.autolog(every_n_iter=1)

def train_test(params):
    start_time = time.time()
    
    with mlflow.start_run(run_name=params["exp_name"]):
        mlflow.set_tag("name", params["exp_name"])

        mlflow.log_param("number_of_epochs", params["number_of_epochs"])
        mlflow.log_param("learning_rate", params["learning_rate"])
        mlflow.log_param("batch_size", params["batch_size"])
        mlflow.log_param("reduceLrPatience", params["reduceLrPatience"])
        mlflow.log_param("reduceLrFactor", params["reduceLrFactor"])
        mlflow.log_param("dropout1", params['dropout1'])
        mlflow.log_param("dropout2", params['dropout2'])
        mlflow.log_param("fc1", params['fc1'])
        mlflow.log_param("fc2", params['fc2'])
        mlflow.log_param("fc3", params['fc3'])
        mlflow.log_param("l2", params['l2'])
        mlflow.log_metric("my ", 1111)

        model = get_model(params)

        optimizer = Adam(lr=params["learning_rate"], decay=params["decay"])  

        model.compile(
          optimizer=optimizer, 
          loss='binary_crossentropy', 
          metrics=['acc']
        )

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', mode='min', patience=params["reduceLrPatience"], factor=params["reduceLrFactor"], min_lr=0.0000001, verbose=2
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', mode='min', patience=params["earlyStoppingPatience"], restore_best_weights=True, verbose=2
            ),
            tf.keras.callbacks.TerminateOnNaN()
        ]

        with converter_train.make_tf_dataset() as train_dataset, converter_test.make_tf_dataset() as test_dataset:

            train_dataset = (train_dataset
                .unbatch()
                .batch(params["batch_size"])
                .map(lambda x: (
                    random_contrast_brightness(basic_transform(x.image)),  
                    tf.reshape(tf.cast(x.labels, dtype=tf.uint8), (-1, num_classes))
                ))
            )

            test_dataset = (test_dataset
                .unbatch()
                .batch(params["batch_size"])
                .map(lambda x: (
                    basic_transform(x.image),  
                    tf.reshape(tf.cast(x.labels, dtype=tf.uint8), (-1, 14))
                ))
            )

            model.fit(
              train_dataset, steps_per_epoch=int(train_rows / params["batch_size"]),  
              epochs=params["number_of_epochs"], verbose=2, 
              validation_data=test_dataset, validation_steps=int(test_rows / params["batch_size"]), 
              callbacks=callbacks
            )

            end_time = int(time.time())
            mlflow.log_metric("training_duration", (end_time-start_time))

            mlflow.set_tag('candidate', 'true')


# COMMAND ----------

# DBTITLE 1,Parameters
params = {
    'exp_name': 'train-01',
  
    'number_of_epochs': 1,
    'batch_size':64,

    'earlyStoppingPatience':14,
    'reduceLrPatience':2,
    'reduceLrFactor':0.4  ,
    'decay':0.0000001,
  
    'learning_rate':	0.0006767142302381941,
    'range1':	0.17551736343366015,
    'range2':	0.1044912957603954,
    'range3':	0.21763139257768388,
  
    'decay':	1e-05,
    'dropout_before':	0.06843850545326827,

    'dropout1':	0.06417977600217108,
    'dropout2':	0.04197058318750846,
    'fc1':	136.0,
    'fc2':	40.0,
    'fc3':	64.0,
    'l2':	0.00030270096150241115,
}

# COMMAND ----------

from  mlflow.tracking import MlflowClient
train_test(params)

# COMMAND ----------

dbutils.notebook.exit("Done")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC # Distributed hyperparameter search
# MAGIC ## Hyperopt & Horovod

# COMMAND ----------

# DBTITLE 1,Define search space
from hyperopt import fmin, rand, tpe, hp, Trials, exceptions, space_eval, STATUS_FAIL, STATUS_OK

search_space = {
  "learning_rate": hp.uniform('learning_rate', 0.0001, 0.005),
  "decay": hp.uniform('decay', 0.000001, 0.00001),
  "l2": hp.uniform('l2', 0.0003, 0.01),
  "reduceLrFactor":hp.uniform('reduceLrFactor', 0.1, 0.5), 
  "reduceLrPatience":hp.quniform('reduceLrPatience', 2, 6, 1),
   
  "dropout1":hp.uniform('dropout1', 0.001, 0.3),
  "dropout2":hp.uniform('dropout2', 0.0001, 0.3),
  "fc1": hp.quniform('fc1', 32, 256, 8),
  "fc2": hp.quniform('fc2', 12, 128, 8),
  "fc3": hp.quniform('fc3', 12, 96, 4),
  
}

# COMMAND ----------

# DBTITLE 1,Define HyperOpt function, which will be called for each experiment
def train_hyper_opt(params):
  params['exp_name'] = 'hyperopt_9'
  
  numberGpus = 8
  number_of_epochs=2 
  earlyStoppingPatience=10

  params['earlyStoppingPatience']=earlyStoppingPatience
  params['numberGpus']=numberGpus
  params['number_of_epochs']=number_of_epochs
  
  params['batch_size']=96
  
  hr = HorovodRunner(np=numberGpus) 
  res = hr.run(train_hvd, params=params)
  res['status'] = STATUS_OK
  return  {'loss': -res['val_roc_auc'], 'status': STATUS_OK}

# COMMAND ----------

# DBTITLE 1,Run Hyperparameter Optimisation using HyperOpt
import hyperopt
argmin = fmin(
  fn=train_hyper_opt,
  space=search_space,
  algo=hyperopt.atpe.suggest,
  max_evals=5,
  show_progressbar=False)

# COMMAND ----------

argmin

# COMMAND ----------


