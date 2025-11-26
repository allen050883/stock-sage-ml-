# deploy/sagemaker_deploy.py
import sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker import get_execution_role
import boto3
import os

def deploy_to_sagemaker(bucket=None, role=None):
    """
    部署到SageMaker
    """
    
    # 1. 設置
    try:
        role = role or get_execution_role()
        sess = sagemaker.Session()
        bucket = bucket or sess.default_bucket()
    except Exception as e:
        print(f"SageMaker session setup failed. Please check your AWS credentials and IAM role.")
        print(f"Error: {e}")
        print("You can manually provide the 'role' and 'bucket' arguments.")
        return None, None

    # 定義S3前綴
    s3_prefix = 'tw-stock-prediction'

    # 定義本地數據路徑
    local_data_path = 'data/processed'
    
    # 2. 上傳數據到S3
    print(f"Uploading data to s3://{bucket}/{s3_prefix}/data")
    train_s3 = sess.upload_data(
        path=os.path.join(local_data_path, 'train.csv'),
        bucket=bucket,
        key_prefix=f'{s3_prefix}/data/train'
    )
    test_s3 = sess.upload_data(
        path=os.path.join(local_data_path, 'test.csv'),
        bucket=bucket,
        key_prefix=f'{s3_prefix}/data/test'
    )
    print("Data upload complete.")
    
    # 3. 創建TensorFlow estimator
    estimator = TensorFlow(
        entry_point='train.py',
        source_dir='../models',  # 相對路徑
        role=role,
        instance_count=1,
        instance_type='ml.p3.2xlarge',  # GPU instance
        framework_version='2.12',
        py_version='py310',
        hyperparameters={
            'sequence-length': 60,
            'epochs': 100,
            'batch-size': 32,
            'use-sagemaker': 'True' # Pass as string
        }
    )
    
    # 4. 訓練
    print("Starting SageMaker training job...")
    estimator.fit({
        'train': train_s3,
        'test': test_s3
    })
    print("Training job complete.")
    
    # 5. 部署
    print("Deploying model to a SageMaker endpoint...")
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.xlarge'
    )
    print(f"Endpoint '{predictor.endpoint_name}' deployed successfully.")
    
    return predictor, estimator

def download_model_from_s3(estimator, download_path='saved_model/'):
    """
    從S3下載訓練好的模型
    """
    if not estimator or not hasattr(estimator, 'model_data'):
        print("Invalid estimator object. Cannot download model.")
        return

    s3 = boto3.client('s3')
    model_data_s3_path = estimator.model_data
    
    # 解析S3路徑
    bucket = model_data_s3_path.split('/')[2]
    key = '/'.join(model_data_s3_path.split('/')[3:])
    
    # 創建下載目錄
    os.makedirs(download_path, exist_ok=True)
    local_model_path = os.path.join(download_path, 'model.tar.gz')

    # 下載
    print(f"Downloading model from s3://{bucket}/{key} to {local_model_path}")
    s3.download_file(bucket, key, local_model_path)
    
    # 解壓
    import tarfile
    print(f"Extracting model to {download_path}")
    with tarfile.open(local_model_path) as tar:
        tar.extractall(path=download_path)
    
    # 清理壓縮檔
    os.remove(local_model_path)
    
    print(f"模型已下載並解壓到 {download_path} 目錄")

if __name__ == '__main__':
    # 執行此腳本需要配置好AWS環境
    # 注意：執行 deploy_to_sagemaker 會產生AWS費用
    print("Running SageMaker deployment script...")
    predictor, estimator = deploy_to_sagemaker()
    
    if estimator:
        # 下載模型
        download_model_from_s3(estimator, download_path='saved_models/from_s3/')
