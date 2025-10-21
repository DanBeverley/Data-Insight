import pytest
import os
from pathlib import Path
import json
import time


pytestmark = pytest.mark.cloud_integration


def azure_credentials_available():
    """Check if Azure credentials are configured"""
    return all([
        os.getenv("AZURE_SUBSCRIPTION_ID"),
        os.getenv("AZURE_RESOURCE_GROUP"),
        os.getenv("AZURE_WORKSPACE_NAME")
    ])


def aws_credentials_available():
    """Check if AWS credentials are configured"""
    return all([
        os.getenv("AWS_ACCESS_KEY_ID"),
        os.getenv("AWS_SECRET_ACCESS_KEY"),
        os.getenv("AWS_REGION")
    ])


@pytest.mark.integration
@pytest.mark.cloud_integration
class TestAzureGPUIntegration:
    """Real Azure ML API integration tests"""

    @pytest.mark.skipif(not azure_credentials_available(), reason="Azure credentials not configured")
    def test_azure_authentication(self):
        """Test Azure ML authentication with real credentials"""
        from azure.identity import DefaultAzureCredential
        from azure.ai.ml import MLClient

        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

        credential = DefaultAzureCredential()
        ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

        assert ml_client is not None
        workspace = ml_client.workspaces.get(workspace_name)
        assert workspace.name == workspace_name

    @pytest.mark.skipif(not azure_credentials_available(), reason="Azure credentials not configured")
    def test_azure_compute_list(self):
        """Test listing Azure ML compute resources"""
        from azure.identity import DefaultAzureCredential
        from azure.ai.ml import MLClient

        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

        credential = DefaultAzureCredential()
        ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

        compute_list = list(ml_client.compute.list())
        assert isinstance(compute_list, list)

    @pytest.mark.skipif(not azure_credentials_available(), reason="Azure credentials not configured")
    def test_azure_job_submission(self):
        """Test submitting a simple training job to Azure ML"""
        from azure.identity import DefaultAzureCredential
        from azure.ai.ml import MLClient, command
        from azure.ai.ml.entities import Environment

        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

        credential = DefaultAzureCredential()
        ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

        job = command(
            code=".",
            command="python -c 'print(\"Test job\")'",
            environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",
            compute="cpu-cluster",
            display_name="integration-test-job"
        )

        submitted_job = ml_client.jobs.create_or_update(job)
        assert submitted_job is not None
        assert submitted_job.name is not None

        ml_client.jobs.cancel(submitted_job.name)

    @pytest.mark.skipif(not azure_credentials_available(), reason="Azure credentials not configured")
    def test_azure_quota_check(self):
        """Test Azure quota checking via Compute Management API"""
        from data_scientist_chatbot.app.core.quota_tracker import quota_tracker

        status = quota_tracker.get_quota_status()

        assert 'azure' in status
        assert 'used' in status['azure']
        assert 'total' in status['azure']
        assert 'available' in status['azure']

    @pytest.mark.skipif(not azure_credentials_available(), reason="Azure credentials not configured")
    def test_azure_model_artifact_download(self):
        """Test downloading model artifacts from Azure ML"""
        from azure.identity import DefaultAzureCredential
        from azure.ai.ml import MLClient

        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

        credential = DefaultAzureCredential()
        ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

        models = list(ml_client.models.list())
        assert isinstance(models, list)


@pytest.mark.integration
@pytest.mark.cloud_integration
class TestAWSGPUIntegration:
    """Real AWS SageMaker API integration tests"""

    @pytest.mark.skipif(not aws_credentials_available(), reason="AWS credentials not configured")
    def test_aws_authentication(self):
        """Test AWS SageMaker authentication with real credentials"""
        import boto3

        region = os.getenv("AWS_REGION", "us-east-1")
        sagemaker = boto3.client('sagemaker', region_name=region)

        response = sagemaker.list_training_jobs(MaxResults=1)
        assert 'TrainingJobSummaries' in response

    @pytest.mark.skipif(not aws_credentials_available(), reason="AWS credentials not configured")
    def test_aws_s3_access(self):
        """Test S3 access for model artifacts"""
        import boto3

        s3 = boto3.client('s3')
        response = s3.list_buckets()

        assert 'Buckets' in response
        assert isinstance(response['Buckets'], list)

    @pytest.mark.skipif(not aws_credentials_available(), reason="AWS credentials not configured")
    def test_aws_sagemaker_list_training_jobs(self):
        """Test listing SageMaker training jobs"""
        import boto3

        region = os.getenv("AWS_REGION", "us-east-1")
        sagemaker = boto3.client('sagemaker', region_name=region)

        response = sagemaker.list_training_jobs(MaxResults=10)
        assert 'TrainingJobSummaries' in response

    @pytest.mark.skipif(not aws_credentials_available(), reason="AWS credentials not configured")
    def test_aws_iam_role_access(self):
        """Test IAM role configuration for SageMaker"""
        import boto3

        iam = boto3.client('iam')
        response = iam.list_roles(MaxItems=1)

        assert 'Roles' in response

    @pytest.mark.skipif(not aws_credentials_available(), reason="AWS credentials not configured")
    def test_aws_quota_check(self):
        """Test AWS quota checking via Service Quotas API"""
        from data_scientist_chatbot.app.core.quota_tracker import quota_tracker

        status = quota_tracker.get_quota_status()

        assert 'aws' in status
        assert 'used' in status['aws']
        assert 'total' in status['aws']
        assert 'available' in status['aws']

    @pytest.mark.skipif(not aws_credentials_available(), reason="AWS credentials not configured")
    def test_aws_sagemaker_instance_types(self):
        """Test querying available SageMaker instance types"""
        import boto3

        region = os.getenv("AWS_REGION", "us-east-1")
        pricing = boto3.client('pricing', region_name='us-east-1')

        response = pricing.get_products(
            ServiceCode='AmazonSageMaker',
            Filters=[
                {'Type': 'TERM_MATCH', 'Field': 'productFamily', 'Value': 'ML Instance'}
            ],
            MaxResults=1
        )

        assert 'PriceList' in response


@pytest.mark.integration
@pytest.mark.cloud_integration
class TestGPUTrainingFallback:
    """Test fallback logic between Azure and AWS"""

    @pytest.mark.skipif(
        not (azure_credentials_available() and aws_credentials_available()),
        reason="Both Azure and AWS credentials required"
    )
    def test_azure_to_aws_fallback_logic(self):
        """Test fallback from Azure to AWS when Azure fails"""
        from data_scientist_chatbot.app.core.quota_tracker import quota_tracker

        azure_status = quota_tracker.get_quota_status().get('azure', {})
        aws_status = quota_tracker.get_quota_status().get('aws', {})

        if not azure_status.get('available'):
            assert aws_status.get('available'), "AWS should be available when Azure is not"

    @pytest.mark.skipif(
        not (azure_credentials_available() or aws_credentials_available()),
        reason="At least one cloud provider credentials required"
    )
    def test_quota_availability_check(self):
        """Test quota availability checking across providers"""
        from data_scientist_chatbot.app.core.quota_tracker import quota_tracker

        if azure_credentials_available():
            azure_available = quota_tracker.has_available_quota('azure')
            assert isinstance(azure_available, bool)

        if aws_credentials_available():
            aws_available = quota_tracker.has_available_quota('aws')
            assert isinstance(aws_available, bool)


@pytest.mark.integration
@pytest.mark.cloud_integration
class TestEndToEndGPUTraining:
    """End-to-end GPU training workflow tests"""

    @pytest.mark.skipif(not azure_credentials_available(), reason="Azure credentials not configured")
    @pytest.mark.timeout(300)
    def test_simple_model_training_azure(self):
        """Test simple model training on Azure (minimal compute)"""
        from data_scientist_chatbot.app.tools import train_on_azure_gpu

        simple_training_code = """
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'test_model.pkl')
print("MODEL_SAVED:test_model.pkl")
print("Training complete")
"""

        result = train_on_azure_gpu(
            simple_training_code,
            "integration_test_session",
            {"instance_type": "Standard_D2_v2", "timeout": 180}
        )

        assert result is not None
        assert 'stdout' in result or 'stderr' in result

    @pytest.mark.skipif(not aws_credentials_available(), reason="AWS credentials not configured")
    @pytest.mark.timeout(300)
    def test_simple_model_training_aws(self):
        """Test simple model training on AWS (minimal compute)"""
        from data_scientist_chatbot.app.tools import train_on_aws_gpu

        simple_training_code = """
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'test_model.pkl')
print("MODEL_SAVED:test_model.pkl")
print("Training complete")
"""

        result = train_on_aws_gpu(
            simple_training_code,
            "integration_test_session",
            {"instance_type": "ml.m5.large", "timeout": 180}
        )

        assert result is not None
        assert 'stdout' in result or 'stderr' in result
