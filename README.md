# 台股價格預測系統 (SageMaker 專案)

本專案旨在利用機器學習（LSTM / XGBoost）模型預測台股價格走勢，並展示如何將整個工作流程部署到 AWS SageMaker 上，從數據處理、模型訓練到最終的推論端點部署。

---

## **專案架構**

```
.
├── config.yaml               # 專案配置文件
├── data/
│   ├── fetch_stock_data.py   # 下載台股數據
│   └── preprocess.py         # 特徵工程與數據處理
├── deploy/
│   ├── inference.py          # 本地推論腳本
│   └── sagemaker_deploy.py   # SageMaker 訓練與部署腳本
├── models/
│   ├── lstm_model.py         # LSTM 模型定義
│   ├── train.py              # 主要訓練腳本
│   └── xgboost_model.py      # XGBoost 模型定義
├── notebooks/                # Jupyter notebooks 用於實驗
├── README.md                 # 本文件
├── requirements.txt          # Python 依賴套件
├── results/                  # 保存預測結果
├── saved_models/             # 保存本地訓練的模型
└── utils/                    # 工具函數 (回測、視覺化)
```

---

## **步驟一：環境準備 (本地與AWS)**

### **1. 本地環境**

- Python 3.9+
- Git
- AWS CLI (已設定好 `aws configure`)

首先，複製專案並安裝所需的 Python 套件：
```bash
git clone <your-repository-url>
cd stock-sage-ml
pip install -r requirements.txt
```

### **2. AWS 環境**

#### a. **AWS 帳戶與權限**
您需要一個 AWS 帳戶。為了讓 SageMaker 能夠存取 S3 和其他服務，您必須創建一個 IAM Role。

1.  登入 AWS IAM 主控台。
2.  在左側選擇「角色 (Roles)」，然後點擊「建立角色 (Create role)」。
3.  選擇 `AWS service` 作為信任的實體類型。
4.  在「使用案例 (Use case)」下，選擇 `SageMaker`，然後點擊「下一步」。
5.  附加 `AmazonSageMakerFullAccess` 和 `AmazonS3FullAccess` 策略（在生產環境中，應遵循最小權限原則）。
6.  為角色命名（例如 `SageMaker-Stock-ExecutionRole`），然後創建它。
7.  **記下這個角色的 ARN**，格式為 `arn:aws:iam::ACCOUNT_ID:role/YourRoleName`。

#### b. **設定 SageMaker 執行角色**
開啟 `deploy/sagemaker_deploy.py` 檔案，並將您剛剛創建的 IAM Role ARN 更新到 `deploy_to_sagemaker` 函數中。如果您的本地 AWS CLI 設定檔的角色有足夠權限，`get_execution_role()` 會自動尋找，但明確指定更為保險。

```python
# deploy/sagemaker_deploy.py

def deploy_to_sagemaker(bucket=None, role=None):
    """
    部署到SageMaker
    """
    # 最佳實踐：明確指定您的角色 ARN
    role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMaker-Stock-ExecutionRole" 
    
    try:
        # role = role or get_execution_role() # 或者使用自動尋找
        sess = sagemaker.Session()
        # ...
```

---

## **步驟二：本地數據準備與測試**

在將任務推送到雲端之前，先在本地驗證流程是個好習慣。

### **1. 下載數據**
執行 `fetch_stock_data.py` 來下載台股歷史數據。數據會被儲存，但此腳本主要用於驗證 API。
```bash
python data/fetch_stock_data.py
```

### **2. 數據預處理**
這一步會將原始數據轉換為模型可以使用的特徵。您需要自行編寫一個主腳本來調用 `preprocess.py` 中的函數，並將處理好的 `train.csv` 和 `test.csv` 儲存到 `data/processed/` 目錄下。（目前專案範本中未提供此主腳本，需用戶根據需求實現）

### **3. 本地訓練 (可選)**
您可以執行本地訓練來快速驗證模型和訓練腳本是否正常工作。
```bash
python models/train.py --train-path data/processed/train.csv --test-path data/processed/test.csv --model-dir saved_models
```
訓練完成後，模型將保存在 `saved_models/1/` 目錄下。

---

## **步驟三：在 SageMaker 上進行訓練與部署**

這是專案的核心步驟，我們會使用 `sagemaker_deploy.py` 腳本來自動化雲端操作。

### **1. 執行部署腳本**
此腳本會完成以下所有事情：
1.  將 `data/processed/` 下的訓練和測試數據上傳到 S3。
2.  啟動一個 SageMaker 訓練任務 (Training Job)，使用 GPU 實例進行訓練。
3.  訓練完成後，將訓練好的模型部署到一個 SageMaker 端點 (Endpoint)，使其可用於即時推論。

```bash
python deploy/sagemaker_deploy.py
```
**注意：此命令會產生 AWS 費用！** 訓練和部署都會根據您選擇的實例類型計費。

### **2. 檢查結果**
腳本執行成功後，您可以登入 AWS SageMaker 主控台：
- 在「訓練 (Training)」>「訓練任務 (Training jobs)」中找到您的訓練紀錄。
- 在「推論 (Inference)」>「端點 (Endpoints)」中找到已部署的、狀態為 `InService` 的端點。記下端點名稱。

---

## **步驟四：調用 SageMaker 端點進行預測**

端點部署好之後，您的應用程式或任何 AWS SDK 客戶端都可以調用它來獲取預測。

以下是一個 Python 範例，展示如何使用 `boto3` 調用端點：

```python
import boto3
import json
import numpy as np

# --- 設定 ---
ENDPOINT_NAME = "your-sagemaker-endpoint-name" # 從 SageMaker 主控台複製您的端點名稱
REGION_NAME = "us-east-1" # 您的 AWS 區域

# 準備要預測的數據
# 格式必須與訓練時完全相同！
# 這裡是一個假的範例數據，維度為 (1, 60, 20)
# 分別代表 (批次大小, 時間序列長度, 特徵數量)
dummy_data = np.random.rand(1, 60, 20).tolist()

# 創建 SageMaker runtime 客戶端
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=REGION_NAME)

# 發送請求
response = sagemaker_runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=json.dumps(dummy_data)
)

# 解析結果
result = json.loads(response["Body"].read().decode())
predicted_price = result['predictions'][0][0]

print(f"預測的價格是: {predicted_price:.2f}")
```

---

## **步驟五：清理資源 (非常重要！)**

為了避免產生不必要的費用，請在完成實驗後務必刪除 SageMaker 相關資源。您可以透過網頁介面或 AWS CLI 來完成。

### **方法一：透過 AWS 管理主控台 (網頁介面)**

這是最直觀的方式。請依照以下順序刪除資源，以避免依賴性錯誤。

1.  **刪除端點 (Endpoint)**
    *   登入 [AWS SageMaker 主控台](https://console.aws.amazon.com/sagemaker/)。
    *   在左側導覽列中，找到 **推論 (Inference)** > **端點 (Endpoints)**。
    *   勾選您要刪除的端點 (例如 `tensorflow-training-YYYY-MM-DD-HH-MM-SS-XXX`)。
    *   點擊 **動作 (Actions)** > **刪除 (Delete)**。
    *   在彈出的對話框中確認刪除。

2.  **刪除端點組態 (Endpoint Configuration)**
    *   在左側導覽列中，找到 **推論 (Inference)** > **端點組態 (Endpoint configurations)**。
    *   勾選與您剛才刪除的端點同名的組態。
    *   點擊 **動作 (Actions)** > **刪除 (Delete)** 並確認。

3.  **刪除模型 (Model)**
    *   在左側導覽列中，找到 **推論 (Inference)** > **模型 (Models)**。
    *   勾選與您訓練任務相關的模型。
    *   點擊 **動作 (Actions)** > **刪除 (Delete)** 並確認。

4.  **(可選) 刪除 S3 數據**
    *   登入 [AWS S3 主控台](https://s3.console.aws.amazon.com/s3/)。
    *   找到 SageMaker 用於儲存數據的儲存桶 (Bucket)。
    *   找到名為 `tw-stock-prediction` 的資料夾（或您在腳本中定義的前綴）。
    *   選取並刪除該資料夾。

### **方法二：透過 AWS CLI (命令列介面)**

如果您安裝了 AWS CLI，可以使用以下指令快速刪除。

1.  **刪除端點**
    ```bash
    aws sagemaker delete-endpoint --endpoint-name your-sagemaker-endpoint-name
    ```

2.  **刪除端點配置**
    ```bash
    aws sagemaker delete-endpoint-config --endpoint-config-name your-endpoint-config-name
    ```

3.  **刪除模型**
    ```bash
    aws sagemaker delete-model --model-name your-model-name
    ```

**提醒**：您可以在 SageMaker 主控台的「推論」部分找到這些資源的確切名稱。訓練任務本身也會在列表中保留紀錄，但已完成的訓練任務通常不會持續產生費用。
