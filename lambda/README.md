# AWS Lambda Prediction Handler

## Quick Start

This directory contains the AWS Lambda function for serverless ML model predictions.

### Files
- `prediction_handler.py` - Lambda function implementation
- `requirements.txt` - Python dependencies

### Local Testing

```python
import json
from prediction_handler import lambda_handler

# Test event
event = {
    "model_s3_uri": "s3://my-bucket/models/model.pkl",
    "data_s3_uri": "s3://my-bucket/data/input.csv",
    "output_s3_uri": "s3://my-bucket/results/predictions.csv",
    "prediction_column_name": "predicted_value",
    "feature_columns": ["feature1", "feature2", "feature3"]
}

# Invoke handler (requires AWS credentials configured)
response = lambda_handler(event, {})
print(json.dumps(response, indent=2))
```

### Packaging for Deployment

```bash
# Run packaging script from project root
./scripts/cloud/package_lambda.sh

# Output: dist/lambda_deployment.zip (ready for upload)
```

### Deployment

#### AWS Console
1. Go to AWS Lambda console
2. Create function (Python 3.9+ runtime)
3. Upload `dist/lambda_deployment.zip`
4. Set handler: `prediction_handler.lambda_handler`
5. Configure memory (512MB+) and timeout (30s+)

#### AWS CLI
```bash
aws lambda create-function \
  --function-name ml-prediction-handler \
  --runtime python3.9 \
  --handler prediction_handler.lambda_handler \
  --role arn:aws:iam::ACCOUNT_ID:role/lambda-s3-role \
  --zip-file fileb://dist/lambda_deployment.zip \
  --memory-size 512 \
  --timeout 30
```

### Testing

```bash
# Run unit tests
pytest tests/unit/test_lambda_handler.py -v

# Expected: 20 passed, 0 failed
```

### Documentation

See `docs/lambda_prediction_handler.md` for:
- Complete architecture guide
- Deployment instructions
- IAM permissions required
- Performance tuning
- Cost estimation
- Troubleshooting

### Requirements

**AWS Services:**
- S3 (model storage, input data, output results)
- Lambda (serverless compute)
- IAM (permissions)
- CloudWatch (logging and monitoring)

**Python Dependencies:**
- pandas 2.0.3
- scikit-learn 1.3.0
- joblib 1.3.2
- numpy 1.24.3
- boto3 1.28.25

### Event Schema

```json
{
  "model_s3_uri": "s3://bucket/path/to/model.pkl",
  "data_s3_uri": "s3://bucket/path/to/data.csv",
  "output_s3_uri": "s3://bucket/path/to/output.csv",
  "prediction_column_name": "predicted_price",
  "feature_columns": ["sqft", "bedrooms", "bathrooms"]
}
```

### Response Schema

**Success:**
```json
{
  "statusCode": 200,
  "body": {
    "success": true,
    "predictions_generated": 1000,
    "output_s3_uri": "s3://bucket/path/to/output.csv",
    "prediction_column_name": "predicted_price"
  }
}
```

**Error:**
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

### IAM Policy Required

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
        "arn:aws:s3:::your-model-bucket/*",
        "arn:aws:s3:::your-data-bucket/*",
        "arn:aws:s3:::your-results-bucket/*"
      ]
    }
  ]
}
```

### Monitoring

Check CloudWatch Logs for execution details:
- Model download status
- Data processing progress
- Prediction generation
- Upload completion
- Error messages

### Support

For issues or questions, see:
- `docs/lambda_prediction_handler.md` - Complete documentation
- `tests/unit/test_lambda_handler.py` - Test examples
- `dev/implemented/task-4-lambda-handler.md` - Implementation details
