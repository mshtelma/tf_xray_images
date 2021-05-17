# Databricks notebook source
new_cluster_config = """
{
    "spark_version": "8.1.x-scala2.12",
    "node_type_id": "i3.xlarge",
    "aws_attributes": {
      "availability": "ON_DEMAND"
    },
    "num_workers": 2
}
"""

existing_cluster_id = '0414-075331-angle420'
notebook_path = '/Repos/michael.shtelma@databricks.com/tf_xray_images_cicd/integration_tests/tests'  
cicd_repo_id="2147458529668789"
prod_repo_id="2147458529668839"

# COMMAND ----------

import json
import time

from databricks_cli.configure.config import _get_api_client
from databricks_cli.configure.provider import EnvironmentVariableConfigProvider
from databricks_cli.sdk import JobsService

config = EnvironmentVariableConfigProvider().get_config()
api_client = _get_api_client(config, command_name="cicdtemplates-")

# Let's update our Repo to the latest git revision
res = api_client.perform_query('PATCH','/repos/{repo_id}'.format(repo_id=cicd_repo_id), {"branch":"master"})
print(res)

#Now we can run our intergration test job
jobs_service = JobsService(api_client)

notebook_task = {'notebook_path': notebook_path}
#new_cluster = json.loads(new_cluster_config)
res = jobs_service.submit_run(run_name="xxx", existing_cluster_id=existing_cluster_id,  notebook_task=notebook_task, )
run_id = res['run_id']
print(run_id)
result_state = None
while True:
    status = jobs_service.get_run(run_id)
    print(status)
    result_state = status["state"].get("result_state", None)
    if result_state:
        print(result_state)
        break
    else:
        time.sleep(5)
        
assert result_state == "SUCCESS"

if result_state == "SUCCESS":
  api_client.perform_query('PATCH','/repos/{repo_id}'.format(repo_id=prod_repo_id), {"branch":"master"})
