# Databricks notebook source
# here comes tests

# COMMAND ----------

import unittest
class SampleJobIntegrationTest(unittest.TestCase):
    def setUp(self):
      pass


    def test_sample(self):

        
        output_count = spark.range(100).count()
        self.assertGreater(output_count, 0)

    def tearDown(self):
        pass


# COMMAND ----------

suite = unittest.TestSuite()
suite.addTest(SampleJobIntegrationTest('test_sample'))
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)

# COMMAND ----------


