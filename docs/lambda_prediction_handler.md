# AWS Lambda Prediction Handler

## Overview

The AWS Lambda prediction handler enables serverless ML model predictions with automatic scaling and pay-per-use pricing. This infrastructure allows users to run predictions without managing servers.

## Architecture

```
┌─────────────────┐
│   S3 Bucket     │
│  (Model Store)  │
└────────┬────────┘
         │
         │ download model
         ▼
┌─────────────────────────┐
│   Lambda Function       │
│  prediction_handler.py  │
│                         │
│  1. Download model      │
│  2. Download data       │
│  3. Load model          │
│  4. Make predictions    │
│  5. Upload results      │
└────────┬────────────────┘
         │
         │ upload results
         ▼
┌─────────────────┐
│   S3 Bucket     │
│ (Results Store) │
└─────────────────┘
```

## Components

### 1. Lambda Handler (`lambda/prediction_handler.py`)

Production-ready AWS Lambda function with:
- S3 model and data download
- Model loading with joblib
- Prediction generation
- Result upload to S3
- Comprehensive error handling
- Proper HTTP status codes (400 for validation, 500 for system errors)

**Event Payload Schema:**
```json
{
  "model_s3_uri": "s3://bucket/path/to/model.pkl",
  "data_s3_uri": "s3://bucket/path/to/data.csv",
  "output_s3_uri": "s3://bucket/path/to/output.csv",
  "prediction_column_name": "predicted_value",
  "feature_columns": ["feature1", "feature2", "feature3"]
}
```

**Response Schema:**
```json
{
  "statusCode": 200,
  "body": {
    "success": true,
    "predictions_generated": 1000,
    "output_s3_uri": "s3://bucket/path/to/output.csv",
    "prediction_column_name": "predicted_value"
  }
}
```

**Error Response:**
```json
{
  "statusCode": 400,
  "body": {
    "success": false,
    "error": "Missing required feature columns: feature_x",
    "error_type": "ValidationError"
  }
}
```

### 2. Dependencies (`lambda/requirements.txt`)

```
pandas==2.0.3
scikit-learn==1.3.0
joblib==1.3.2
numpy==1.24.3
boto3==1.28.25
```

These versions are compatible with AWS Lambda Python 3.9+ runtime.

### 3. Packaging Script (`scripts/cloud/package_lambda.sh`)

Automated deployment packaging with:
- Dependency installation to `dist/package/`
- Lambda handler inclusion
- ZIP file creation for AWS Lambda
- Validation of required libraries
- Deployment instructions

**Usage:**
```bash
./scripts/cloud/package_lambda.sh
```

**Output:**
- `dist/lambda_deployment.zip` (74MB, 9099 files)
- Ready for AWS Lambda deployment

## Testing

### Test Coverage: 20/20 tests passing

**Test Suite:** `tests/unit/test_lambda_handler.py`

**Coverage Areas:**
1. **S3 URI Parsing (5 tests)**
   - Valid URI parsing
   - Nested path handling
   - Invalid format detection
   - Missing bucket/key validation

2. **Lambda Handler Functionality (12 tests)**
   - Successful prediction workflow
   - Model download from S3
   - Data download from S3
   - Prediction generation
   - Output upload to S3
   - Error handling (missing model, invalid data, S3 failures)
   - Missing required fields
   - Feature column selection
   - Prediction column addition
   - Lambda response format

3. **Edge Cases (3 tests)**
   - Empty dataframe handling
   - Missing feature columns
   - Corrupted model file

**Run Tests:**
```bash
pytest tests/unit/test_lambda_handler.py -v
```

**Expected Output:**
```
20 passed, 6 warnings in 0.17s
```

## Deployment

### Option 1: AWS Console

1. Navigate to AWS Lambda console
2. Create new function or select existing
3. Upload `dist/lambda_deployment.zip`
4. Set handler: `prediction_handler.lambda_handler`
5. Set runtime: Python 3.9 or 3.10
6. Configure:
   - Memory: 512MB minimum (adjust for model size)
   - Timeout: 30s minimum (adjust for data size)
   - IAM Role: S3 read/write permissions

### Option 2: AWS CLI

```bash
# Create function
aws lambda create-function \
  --function-name ml-prediction-handler \
  --runtime python3.9 \
  --handler prediction_handler.lambda_handler \
  --role arn:aws:iam::ACCOUNT_ID:role/lambda-s3-role \
  --zip-file fileb://dist/lambda_deployment.zip \
  --memory-size 512 \
  --timeout 30

# Update existing function
aws lambda update-function-code \
  --function-name ml-prediction-handler \
  --zip-file fileb://dist/lambda_deployment.zip
```

### Option 3: Terraform

```hcl
resource "aws_lambda_function" "ml_predictions" {
  filename         = "dist/lambda_deployment.zip"
  function_name    = "ml-prediction-handler"
  role            = aws_iam_role.lambda_role.arn
  handler         = "prediction_handler.lambda_handler"
  source_code_hash = filebase64sha256("dist/lambda_deployment.zip")
  runtime         = "python3.9"
  memory_size     = 512
  timeout         = 30
}

resource "aws_iam_role" "lambda_role" {
  name = "lambda-s3-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_s3" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}
```

## IAM Permissions Required

The Lambda function requires the following S3 permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": [
        "arn:aws:s3:::ml-models/*",
        "arn:aws:s3:::ml-data/*",
        "arn:aws:s3:::ml-results/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

## Invocation Examples

### Synchronous Invocation (AWS CLI)

```bash
aws lambda invoke \
  --function-name ml-prediction-handler \
  --payload '{
    "model_s3_uri": "s3://ml-models/user_123/model_rf.pkl",
    "data_s3_uri": "s3://ml-data/user_123/input.csv",
    "output_s3_uri": "s3://ml-results/user_123/predictions.csv",
    "prediction_column_name": "predicted_price",
    "feature_columns": ["sqft", "bedrooms", "bathrooms"]
  }' \
  response.json
```

### Python (boto3)

```python
import boto3
import json

lambda_client = boto3.client('lambda')

payload = {
    "model_s3_uri": "s3://ml-models/user_123/model_rf.pkl",
    "data_s3_uri": "s3://ml-data/user_123/input.csv",
    "output_s3_uri": "s3://ml-results/user_123/predictions.csv",
    "prediction_column_name": "predicted_price",
    "feature_columns": ["sqft", "bedrooms", "bathrooms"]
}

response = lambda_client.invoke(
    FunctionName='ml-prediction-handler',
    InvocationType='RequestResponse',
    Payload=json.dumps(payload)
)

result = json.loads(response['Payload'].read())
print(result)
```

## Performance Considerations

### Memory Configuration

| Model Size | Data Size | Recommended Memory |
|------------|-----------|-------------------|
| < 10MB     | < 1MB     | 512MB            |
| 10-50MB    | 1-10MB    | 1024MB           |
| 50-100MB   | 10-50MB   | 2048MB           |
| > 100MB    | > 50MB    | 3008MB           |

### Timeout Configuration

| Operation          | Typical Duration | Recommended Timeout |
|-------------------|------------------|---------------------|
| Download + Load    | 1-5s            | 30s                |
| Prediction (1K rows)| 0.5-2s         | 30s                |
| Prediction (10K rows)| 2-10s         | 60s                |
| Prediction (100K rows)| 10-30s       | 120s                |

### Cold Start Optimization

**Cold Start Time:** ~3-5 seconds (includes dependency loading)

**Optimization Strategies:**
1. **Provisioned Concurrency:** Pre-warm Lambda instances
2. **Layer Optimization:** Extract heavy dependencies to Lambda layers
3. **Model Size Reduction:** Use model compression techniques

## Error Handling

### Error Categories

**400 - Validation Errors:**
- Missing required event fields
- Invalid S3 URI format
- Model file not found in S3
- Data file not found in S3
- Missing feature columns in data
- Empty dataframe
- Corrupted model file

**500 - System Errors:**
- Unexpected exceptions
- Memory exhaustion
- Timeout errors

### Monitoring

**CloudWatch Metrics:**
- Invocation count
- Error count
- Duration
- Throttles
- Memory usage

**Custom Logging:**
```python
# Lambda handler logs to CloudWatch
print(f"Downloading model from {model_s3_uri}")
print(f"Generating predictions for {len(data)} rows")
print(f"Uploading results to {output_s3_uri}")
```

## Cost Estimation

**Pricing Model:**
- $0.20 per 1M requests
- $0.0000166667 per GB-second

**Example Costs (us-east-1):**

| Scenario | Memory | Duration | Monthly Requests | Cost |
|----------|--------|----------|------------------|------|
| Small    | 512MB  | 5s       | 10,000          | $0.42 |
| Medium   | 1024MB | 10s      | 100,000         | $17.08 |
| Large    | 2048MB | 30s      | 1,000,000       | $1,020.00 |

**Free Tier:**
- 1M requests per month
- 400,000 GB-seconds per month

## Security Best Practices

1. **Least Privilege IAM:** Only grant required S3 permissions
2. **S3 Bucket Policies:** Restrict access to Lambda role
3. **Encryption:** Enable S3 server-side encryption (SSE-S3 or SSE-KMS)
4. **VPC Configuration:** Deploy Lambda in VPC for private S3 access
5. **Environment Variables:** Use AWS Secrets Manager for sensitive data
6. **Input Validation:** Handler validates all input parameters

## Troubleshooting

### Common Issues

**Issue:** "File not found in S3"
- **Cause:** Incorrect S3 URI or insufficient IAM permissions
- **Solution:** Verify URI format and IAM policy

**Issue:** "Memory exhausted"
- **Cause:** Model or data too large for allocated memory
- **Solution:** Increase Lambda memory or reduce data size

**Issue:** "Task timed out"
- **Cause:** Large dataset or slow S3 download
- **Solution:** Increase timeout or optimize data size

**Issue:** "Missing feature columns"
- **Cause:** Feature names don't match training data
- **Solution:** Verify feature_columns list matches model training

## Maintenance

### Updating Dependencies

1. Edit `lambda/requirements.txt`
2. Run packaging script: `./scripts/cloud/package_lambda.sh`
3. Deploy updated ZIP to Lambda

### Updating Handler Logic

1. Edit `lambda/prediction_handler.py`
2. Run tests: `pytest tests/unit/test_lambda_handler.py`
3. Run packaging script
4. Deploy to Lambda

## Future Enhancements

- [ ] Support for multiple file formats (JSON, Parquet)
- [ ] Batch prediction API with pagination
- [ ] Model caching to reduce cold starts
- [ ] Prediction confidence intervals
- [ ] A/B testing support for multiple models
- [ ] CloudWatch dashboard template
- [ ] Integration with AWS Step Functions
- [ ] Real-time predictions via API Gateway

## References

- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [AWS Lambda Python Runtime](https://docs.aws.amazon.com/lambda/latest/dg/lambda-python.html)
- [Boto3 S3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html)
- [scikit-learn Model Persistence](https://scikit-learn.org/stable/model_persistence.html)
