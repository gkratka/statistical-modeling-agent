# Cloud Architecture Documentation

**Version:** 1.0
**Last Updated:** 2025-11-08
**Target Audience:** Software Engineers, System Architects

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Provider Factory Pattern](#2-provider-factory-pattern)
3. [State Management](#3-state-management)
4. [Data Flow](#4-data-flow)
5. [Sequence Diagrams](#5-sequence-diagrams)
6. [Error Handling](#6-error-handling)
7. [Recovery Strategies](#7-recovery-strategies)

---

## 1. System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TELEGRAM BOT INTERFACE                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   Workflow Handlers     │
                    │  (handlers.py)          │
                    └────────────┬────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
    ┌─────▼──────┐      ┌───────▼────────┐     ┌──────▼───────┐
    │   Local    │      │  Cloud         │     │   Cloud      │
    │  Training  │      │  Training      │     │ Prediction   │
    │  Handlers  │      │  Handlers      │     │  Handlers    │
    └─────┬──────┘      └───────┬────────┘     └──────┬───────┘
          │                     │                      │
          │             ┌───────▼────────┐            │
          │             │ Provider       │            │
          │             │ Factory        │            │
          │             └───────┬────────┘            │
          │                     │                      │
          │        ┌────────────┼────────────┐        │
          │        │            │            │        │
    ┌─────▼──────┐ │    ┌──────▼──────┐    │  ┌─────▼────────┐
    │   Local    │ │    │     AWS     │    │  │   RunPod     │
    │ ML Engine  │ │    │  Providers  │    │  │  Providers   │
    └────────────┘ │    └──────┬──────┘    │  └──────┬───────┘
                   │           │            │         │
                   │     ┌─────▼─────┐      │    ┌────▼────┐
                   │     │ S3 Storage│      │    │ Network │
                   │     │ EC2 Train │      │    │ Volume  │
                   │     │ Lambda    │      │    │ GPU Pod │
                   │     └───────────┘      │    └─────────┘
                   │                        │
                   └────────────────────────┘
                           CLOUD LAYER
```

### Component Responsibilities

| Layer | Component | Responsibility |
|-------|-----------|----------------|
| **Interface** | Telegram Bot | User interaction, message routing |
| **Routing** | Workflow Handlers | Data source detection, workflow orchestration |
| **Abstraction** | Provider Factory | Cloud provider selection, health checks, fallback |
| **Implementation** | Cloud Providers | Storage, training, prediction operations |
| **Infrastructure** | Cloud Services | AWS (S3, EC2, Lambda), RunPod (Volumes, Pods, Serverless) |

---

## 2. Provider Factory Pattern

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       CloudProviderFactory                              │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │  Auto-Detection Module                                        │     │
│  │  • detect_available_providers()                               │     │
│  │  • check_provider_health(provider, credentials)               │     │
│  │  • get_best_provider()                                        │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │  Provider Creation Methods                                    │     │
│  │  • create_storage_provider(provider, config)                  │     │
│  │  • create_training_provider(provider, config)                 │     │
│  │  • create_prediction_provider(provider, config)               │     │
│  │  • create_provider_with_fallback(type, config)                │     │
│  └──────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                │                     │                     │
      ┌─────────▼─────────┐  ┌────────▼────────┐  ┌───────▼────────┐
      │  AWS Providers    │  │ RunPod Providers│  │ Local Fallback │
      └─────────┬─────────┘  └────────┬────────┘  └───────┬────────┘
                │                     │                    │
      ┌─────────▼─────────┐  ┌────────▼────────┐  ┌───────▼────────┐
      │ • S3Manager       │  │ • RunPodStorage │  │ • LocalStorage │
      │ • EC2Manager      │  │ • RunPodPod     │  │ • LocalML      │
      │ • LambdaManager   │  │ • RunPodServer  │  │                │
      └───────────────────┘  └─────────────────┘  └────────────────┘
```

### Factory Pattern Implementation

**Class Diagram:**

```
┌─────────────────────────────────────────────────────────────┐
│           <<interface>> CloudStorageProvider                │
├─────────────────────────────────────────────────────────────┤
│ + upload_dataset(user_id, file_path, name): str            │
│ + save_model(user_id, model_id, model_dir): str            │
│ + load_model(user_id, model_id, local_dir): Path           │
│ + list_user_datasets(user_id): list[dict]                  │
│ + list_user_models(user_id): list[dict]                    │
└─────────────────────────────────────────────────────────────┘
                         ▲                  ▲
                         │                  │
           ┌─────────────┴──────┐  ┌────────┴──────────────┐
           │   S3Manager        │  │ RunPodStorageManager  │
           ├────────────────────┤  ├───────────────────────┤
           │ - aws_client       │  │ - config              │
           │ - config           │  │ - s3_client           │
           └────────────────────┘  └───────────────────────┘


┌─────────────────────────────────────────────────────────────┐
│          <<interface>> CloudTrainingProvider                │
├─────────────────────────────────────────────────────────────┤
│ + select_compute_type(...): str                             │
│ + launch_training(config): dict                             │
│ + monitor_training(job_id): dict                            │
│ + terminate_training(job_id): str                           │
└─────────────────────────────────────────────────────────────┘
                         ▲                  ▲
                         │                  │
           ┌─────────────┴──────┐  ┌────────┴──────────────┐
           │   EC2Manager       │  │ RunPodPodManager      │
           ├────────────────────┤  ├───────────────────────┤
           │ - aws_client       │  │ - config              │
           │ - config           │  │ - runpod_client       │
           └────────────────────┘  └───────────────────────┘
```

### Provider Selection Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│ START: create_provider_with_fallback(type='training')               │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                      ┌─────────▼──────────┐
                      │ detect_available_  │
                      │   providers()      │
                      └─────────┬──────────┘
                                │
                   ┌────────────▼────────────┐
                   │ Priority Order:         │
                   │ 1. RunPod               │
                   │ 2. AWS                  │
                   │ 3. Local                │
                   └────────────┬────────────┘
                                │
           ┌────────────────────┼────────────────────┐
           │                    │                    │
    ┌──────▼───────┐    ┌───────▼──────┐    ┌──────▼──────┐
    │   RunPod     │    │     AWS      │    │   Local     │
    │ Available?   │    │ Available?   │    │  Fallback   │
    └──────┬───────┘    └───────┬──────┘    └──────┬──────┘
           │                    │                   │
      ┌────▼────┐          ┌────▼────┐        ┌────▼────┐
      │ Health  │          │ Health  │        │ Always  │
      │ Check   │          │ Check   │        │ Healthy │
      └────┬────┘          └────┬────┘        └────┬────┘
           │                    │                   │
      ┌────▼────┐          ┌────▼────┐        ┌────▼────┐
      │ Create  │          │ Create  │        │ Return  │
      │ RunPod  │    OR    │  AWS    │   OR   │  None   │
      │Provider │          │Provider │        │ (local) │
      └────┬────┘          └────┬────┘        └────┬────┘
           │                    │                   │
           └────────────────────┴───────────────────┘
                                │
                      ┌─────────▼──────────┐
                      │ Return Provider    │
                      │    Instance        │
                      └────────────────────┘
```

### Health Check Algorithm

```python
def check_provider_health(provider: str, **credentials) -> dict:
    """
    Multi-layer health check with timeout.

    Returns:
        {
            'healthy': bool,
            'errors': List[str]  # Empty if healthy
        }
    """
    # Layer 1: Credential Validation
    if not validate_credentials(provider, credentials):
        return {'healthy': False, 'errors': ['Invalid credentials']}

    # Layer 2: Connectivity Test (with timeout)
    try:
        with timeout(5):  # 5 second timeout
            if provider == 'runpod':
                test_runpod_api(credentials['api_key'])
            elif provider == 'aws':
                test_aws_api(credentials)
    except TimeoutError:
        return {'healthy': False, 'errors': ['Connection timeout']}
    except Exception as e:
        return {'healthy': False, 'errors': [str(e)]}

    # Layer 3: Service Availability
    if not check_service_availability(provider):
        return {'healthy': False, 'errors': ['Service unavailable']}

    return {'healthy': True, 'errors': []}
```

---

## 3. State Management

### State Machine Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CLOUD TRAINING STATE MACHINE                    │
└─────────────────────────────────────────────────────────────────────┘

  START (/train command)
    │
    ▼
┌────────────────────────────┐
│ CHOOSING_TRAINING_LOCATION │
│ Decision: Local or Cloud?  │
└────────────┬───────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌─────────┐    ┌──────────────┐
│ LOCAL   │    │ CLOUD        │
│TRAINING │    │ WORKFLOWS    │
└─────────┘    └──────┬───────┘
                      │
                      ▼
            ┌──────────────────┐
            │ AWAITING_S3_     │
            │   DATASET        │
            │ Input: Telegram/ │
            │  Path/S3/RunPod  │
            └──────┬───────────┘
                   │
          ┌────────┴─────────┐
          │                  │
          ▼                  ▼
    ┌───────────┐    ┌──────────────┐
    │Auto-Detect│    │Manual Schema │
    │  Schema   │    │    Entry     │
    └─────┬─────┘    └──────┬───────┘
          │                 │
          └────────┬────────┘
                   │
                   ▼
          ┌──────────────┐
          │ CONFIRMING_  │
          │   SCHEMA     │
          │ Review & OK? │
          └──────┬───────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
    ┌────────┐      ┌─────────┐
    │ Accept │      │ Reject  │
    └────┬───┘      └────┬────┘
         │               │
         │       ┌───────┘
         │       │
         ▼       ▼
    ┌──────────────────┐
    │ SELECTING_MODEL_ │
    │   CATEGORY       │
    │ Reg/Class/Neural │
    └──────┬───────────┘
           │
           ▼
    ┌──────────────┐
    │ SELECTING_   │
    │   MODEL      │
    │ Choose Model │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ CONFIRMING_  │
    │   MODEL      │
    │Hyperparameters│
    └──────┬───────┘
           │
           ▼
    ┌──────────────────┐
    │ CONFIRMING_      │
    │ INSTANCE_TYPE    │
    │ GPU & Cost OK?   │
    └──────┬───────────┘
           │
           ▼
    ┌──────────────┐
    │ LAUNCHING_   │
    │  TRAINING    │
    │ Pod Creation │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ MONITORING_  │
    │  TRAINING    │
    │ Log Streaming│
    └──────┬───────┘
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
┌─────────┐  ┌────────┐
│SUCCESS  │  │ ERROR  │
└────┬────┘  └───┬────┘
     │           │
     ▼           ▼
┌──────────────────┐
│ TRAINING_        │
│  COMPLETE        │
│Results & Cleanup │
└──────────────────┘
```

### State Transition Rules

| From State | To State | Trigger | Validation |
|------------|----------|---------|------------|
| `CHOOSING_TRAINING_LOCATION` | `AWAITING_S3_DATASET` | User: "Cloud" | None |
| `AWAITING_S3_DATASET` | `CHOOSING_LOAD_STRATEGY` | Dataset uploaded | File format valid, size OK |
| `CHOOSING_LOAD_STRATEGY` | `CONFIRMING_SCHEMA` | User: "Auto" | Dataset loaded successfully |
| `CONFIRMING_SCHEMA` | `SELECTING_MODEL_CATEGORY` | User: "Accept" | Schema has target + features |
| `SELECTING_MODEL` | `CONFIRMING_MODEL` | User selects model | Model available for task type |
| `CONFIRMING_INSTANCE_TYPE` | `LAUNCHING_TRAINING` | User: "Launch" | Budget not exceeded |
| `MONITORING_TRAINING` | `TRAINING_COMPLETE` | Training done | Model saved to storage |
| Any State | `ERROR_STATE` | Exception | Error logged |

### State Persistence

**Database Schema:**

```sql
CREATE TABLE workflow_states (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    conversation_id VARCHAR(255) NOT NULL,
    state VARCHAR(100) NOT NULL,
    state_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    UNIQUE(user_id, conversation_id)
);

CREATE INDEX idx_user_conversation ON workflow_states(user_id, conversation_id);
CREATE INDEX idx_expires_at ON workflow_states(expires_at);
```

**State Data Example:**

```json
{
  "state": "CONFIRMING_INSTANCE_TYPE",
  "workflow_type": "CLOUD_TRAINING",
  "data": {
    "dataset_uri": "s3://ml-agent-data-123/datasets/user_456/housing.csv",
    "schema": {
      "target_column": "price",
      "feature_columns": ["sqft", "bedrooms", "bathrooms"],
      "task_type": "regression"
    },
    "model_config": {
      "model_type": "random_forest",
      "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10
      }
    },
    "compute_config": {
      "provider": "runpod",
      "gpu_type": "NVIDIA RTX A5000",
      "estimated_cost": 0.015,
      "estimated_time_minutes": 3
    }
  },
  "history": [
    {"state": "CHOOSING_TRAINING_LOCATION", "timestamp": "2025-11-08T10:00:00Z"},
    {"state": "AWAITING_S3_DATASET", "timestamp": "2025-11-08T10:00:15Z"},
    {"state": "CONFIRMING_SCHEMA", "timestamp": "2025-11-08T10:00:45Z"},
    {"state": "SELECTING_MODEL_CATEGORY", "timestamp": "2025-11-08T10:01:10Z"},
    {"state": "SELECTING_MODEL", "timestamp": "2025-11-08T10:01:25Z"},
    {"state": "CONFIRMING_MODEL", "timestamp": "2025-11-08T10:01:50Z"},
    {"state": "CONFIRMING_INSTANCE_TYPE", "timestamp": "2025-11-08T10:02:15Z"}
  ]
}
```

### Transaction-Safe State Updates

```python
@transaction
def update_state(user_id: int, conversation_id: str, new_state: str, data: dict):
    """
    Atomic state transition with rollback on failure.

    Steps:
    1. Snapshot current state
    2. Validate transition is allowed
    3. Update state in database
    4. If error, rollback to snapshot
    """
    # 1. Snapshot
    snapshot = get_current_state(user_id, conversation_id)

    try:
        # 2. Validate
        validate_transition(snapshot['state'], new_state)

        # 3. Update
        db.execute("""
            UPDATE workflow_states
            SET state = %s,
                state_data = %s,
                updated_at = NOW()
            WHERE user_id = %s AND conversation_id = %s
        """, (new_state, json.dumps(data), user_id, conversation_id))

        db.commit()

    except Exception as e:
        # 4. Rollback
        db.rollback()
        restore_state(snapshot)
        raise StateTransitionError(f"Failed to transition to {new_state}: {e}")
```

---

## 4. Data Flow

### Training Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING DATA FLOW                           │
└─────────────────────────────────────────────────────────────────────┘

Telegram                Bot              Cloud Storage         Training
  User                Handler              (S3/Volume)          Instance
   │                     │                      │                  │
   │──Upload CSV────────>│                      │                  │
   │                     │──Validate────>       │                  │
   │                     │                      │                  │
   │                     │──Upload Dataset─────>│                  │
   │                     │  (multipart if >50MB)│                  │
   │                     │                      │                  │
   │                     │<─Storage URI────────│                  │
   │                     │  s3://bucket/data    │                  │
   │                     │                      │                  │
   │                     │──Launch Training────────────────────────>│
   │                     │  (pass storage URI)  │                  │
   │                     │                      │                  │
   │                     │                      │<─Download Dataset──│
   │                     │                      │                  │
   │                     │                      │                  │
   │<─Status: Training──│                      │        Training  │
   │                     │                      │        In Progress│
   │                     │<─────────────────Stream Logs─────────────│
   │<─Log Updates───────│                      │                  │
   │                     │                      │                  │
   │                     │                      │<─Save Model────────│
   │                     │                      │  (model.pkl,     │
   │                     │                      │   scaler.pkl,    │
   │                     │                      │   metadata.json) │
   │                     │                      │                  │
   │                     │<─Model URI───────────│                  │
   │                     │  s3://bucket/models/ │                  │
   │                     │                      │                  │
   │                     │──Terminate Instance─────────────────────>│
   │                     │  (CRITICAL: Stop costs)                 │
   │                     │                      │                  │
   │<─Training Complete─│                      │                  │
   │  • Model ID        │                      │                  │
   │  • Metrics         │                      │                  │
   │  • Cost            │                      │                  │
   │  • Storage URI     │                      │                  │
   │                     │                      │                  │
```

### Prediction Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PREDICTION DATA FLOW                           │
└─────────────────────────────────────────────────────────────────────┘

Telegram             Bot              Cloud Storage        Serverless
  User             Handler             (S3/Volume)         Function
   │                  │                     │                  │
   │──/predict_cloud─>│                     │                  │
   │                  │──List Models───────>│                  │
   │                  │<─Model List─────────│                  │
   │                  │                     │                  │
   │<─Model Selection│                     │                  │
   │                  │                     │                  │
   │──Select Model───>│                     │                  │
   │                  │──Validate Model────>│                  │
   │                  │<─Model Valid────────│                  │
   │                  │                     │                  │
   │──Upload Data────>│                     │                  │
   │                  │──Upload to Storage─>│                  │
   │                  │<─Data URI───────────│                  │
   │                  │                     │                  │
   │                  │──Invoke Prediction──────────────────────>│
   │                  │  {                  │                  │
   │                  │    model_uri,       │                  │
   │                  │    data_uri,        │                  │
   │                  │    output_uri       │                  │
   │                  │  }                  │                  │
   │                  │                     │                  │
   │                  │                     │<─Load Model────────│
   │                  │                     │<─Load Data─────────│
   │                  │                     │                  │
   │                  │                     │         Predict  │
   │                  │                     │         (vectorize)│
   │                  │                     │                  │
   │                  │                     │<─Save Results──────│
   │                  │                     │  (predictions.csv)│
   │                  │                     │                  │
   │                  │<─Result URI─────────────────────────────│
   │                  │  s3://bucket/results/                  │
   │                  │                     │                  │
   │                  │──Download Results──>│                  │
   │                  │<─Predictions CSV────│                  │
   │                  │                     │                  │
   │<─Predictions────│                     │                  │
   │  (CSV file +     │                     │                  │
   │   summary stats) │                     │                  │
   │                  │                     │                  │
```

### Multi-Provider Data Flow (with Fallback)

```
┌─────────────────────────────────────────────────────────────────────┐
│                  MULTI-PROVIDER FALLBACK FLOW                       │
└─────────────────────────────────────────────────────────────────────┘

  User Request
      │
      ▼
┌──────────────┐
│ Provider     │
│ Factory      │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Detect Available │
│ Providers        │
└──────┬───────────┘
       │
       ├─────────────────────────────────────────────────┐
       │                                                 │
       ▼                                                 ▼
┌──────────────┐                                ┌────────────────┐
│   RunPod     │                                │      AWS       │
│   Priority 1 │                                │    Priority 2  │
└──────┬───────┘                                └────────┬───────┘
       │                                                 │
       ▼                                                 ▼
┌──────────────┐                                ┌────────────────┐
│ Health Check │                                │  Health Check  │
│  • API Key   │                                │  • AWS Creds   │
│  • Volume ID │                                │  • S3 Access   │
└──────┬───────┘                                └────────┬───────┘
       │                                                 │
       ▼                                                 ▼
    Success?                                         Success?
       │                                                 │
  ┌────┴────┐                                      ┌────┴────┐
  │   YES   │                                      │   YES   │
  └────┬────┘                                      └────┬────┘
       │                                                 │
       ▼                                                 ▼
┌──────────────┐                                ┌────────────────┐
│ Use RunPod   │                                │   Use AWS      │
│ Providers    │                                │   Providers    │
└──────┬───────┘                                └────────┬───────┘
       │                                                 │
       └──────────────────┬──────────────────────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ Execute      │
                   │ Operation    │
                   └──────┬───────┘
                          │
                    ┌─────┴──────┐
                    │            │
              Success         Failure
                    │            │
                    ▼            ▼
            ┌────────────┐  ┌─────────────┐
            │   Return   │  │  Fallback   │
            │   Result   │  │  to Local   │
            └────────────┘  └─────────────┘
```

---

## 5. Sequence Diagrams

### Cloud Training Sequence

```
User     Bot      State        Provider    Storage      Training
         Handler  Manager      Factory                  Instance
 │         │         │            │           │            │
 │─/train─>│         │            │           │            │
 │         │─get_state()          │           │            │
 │         │<─current_state───────│           │            │
 │         │                      │           │            │
 │         │─transition(CHOOSING_TRAINING_LOCATION)        │
 │         │<─state_updated───────│           │            │
 │         │                      │           │            │
 │<─Local/Cloud?                  │           │            │
 │         │                      │           │            │
 │─Cloud──>│                      │           │            │
 │         │─transition(AWAITING_S3_DATASET)  │            │
 │         │                      │           │            │
 │<─Upload CSV                    │           │            │
 │         │                      │           │            │
 │─[file]─>│                      │           │            │
 │         │─validate_dataset()   │           │            │
 │         │                      │           │            │
 │         │─get_best_provider()──────────────>│           │
 │         │<─"runpod"────────────────────────│           │
 │         │                      │           │            │
 │         │─create_storage("runpod")         │           │
 │         │<─storage_provider────────────────│           │
 │         │                      │           │            │
 │         │─upload_dataset()──────────────────────────>   │
 │         │<─storage_uri──────────────────────────────   │
 │         │  (runpod://vol/data.csv)         │           │
 │         │                      │           │            │
 │         │─detect_schema(uri)───────────────────────>   │
 │         │<─schema──────────────────────────────────   │
 │         │                      │           │            │
 │         │─transition(CONFIRMING_SCHEMA)    │           │
 │         │                      │           │            │
 │<─Schema OK?                    │           │            │
 │─Accept─>│                      │           │            │
 │         │                      │           │            │
 │         │─transition(SELECTING_MODEL)      │           │
 │<─Model Categories              │           │            │
 │─Random Forest                  │           │            │
 │         │                      │           │            │
 │         │─transition(CONFIRMING_INSTANCE_TYPE)         │
 │<─GPU: A5000                    │           │            │
 │  Cost: $0.015                  │           │            │
 │─Launch─>│                      │           │            │
 │         │                      │           │            │
 │         │─create_training("runpod")        │           │
 │         │<─training_provider───────────────│           │
 │         │                      │           │            │
 │         │─launch_training(config)──────────────────────>│
 │         │<─job_id──────────────────────────────────────│
 │         │  (pod-xyz123)        │           │            │
 │         │                      │           │            │
 │         │─transition(MONITORING_TRAINING)  │           │
 │         │                      │           │            │
 │         │──────────────────poll_logs()──────────────────>│
 │         │<──────────────────log_stream──────────────────│
 │<─Logs──│                      │           │            │
 │         │                      │           │            │
 │         │                      │           │      [Training Complete]
 │         │<─────────────────job_complete──────────────────│
 │         │                      │           │            │
 │         │─download_metrics()───────────────────────>    │
 │         │<─metrics─────────────────────────────────    │
 │         │                      │           │            │
 │         │─terminate_pod()──────────────────────────────>│
 │         │<─terminated──────────────────────────────────│
 │         │                      │           │            │
 │         │─transition(TRAINING_COMPLETE)    │           │
 │         │                      │           │            │
 │<─Results                       │           │            │
 │  Model ID                      │           │            │
 │  Metrics                       │           │            │
 │  Cost                          │           │            │
 │         │                      │           │            │
```

### Cloud Prediction Sequence

```
User     Bot      State       Provider   Storage    Serverless
         Handler  Manager     Factory               Function
 │         │         │           │          │           │
 │─/predict_cloud   │           │          │           │
 │         │         │           │          │           │
 │         │─get_state()         │          │           │
 │         │<─state──────────────│          │           │
 │         │                     │          │           │
 │<─Model Selection              │          │           │
 │         │                     │          │           │
 │─Model ID                      │          │           │
 │         │                     │          │           │
 │         │─create_storage("runpod")       │           │
 │         │<─storage_provider──────────────│           │
 │         │                     │          │           │
 │         │─list_user_models()─────────────────────>   │
 │         │<─models────────────────────────────────   │
 │         │                     │          │           │
 │<─Upload Data                  │          │           │
 │         │                     │          │           │
 │─[file]─>│                     │          │           │
 │         │─upload_dataset()────────────────────────>  │
 │         │<─data_uri───────────────────────────────  │
 │         │  (runpod://vol/pred.csv)     │           │
 │         │                     │          │           │
 │         │─create_prediction("runpod")    │           │
 │         │<─prediction_provider───────────│           │
 │         │                     │          │           │
 │         │─invoke_prediction({│          │           │
 │         │   model_uri,        │          │           │
 │         │   data_uri,         │          │           │
 │         │   output_uri        │          │           │
 │         │ })──────────────────────────────────────────>│
 │         │                     │          │           │
 │         │                     │          │      [Load Model]
 │         │                     │          │      [Load Data]
 │         │                     │          │      [Predict]
 │         │                     │          │           │
 │         │<─result─────────────────────────────────────│
 │         │  {output_uri, stats}│          │           │
 │         │                     │          │           │
 │         │─download_results()──────────────────────>   │
 │         │<─predictions_csv────────────────────────   │
 │         │                     │          │           │
 │<─Results                      │          │           │
 │  CSV File                     │          │           │
 │  Summary                      │          │           │
 │         │                     │          │           │
```

---

## 6. Error Handling

### Error Hierarchy

```
CloudError (Base Exception)
│
├── CloudConfigurationError
│   ├── MissingCredentialError
│   ├── InvalidCredentialError
│   └── ConfigValidationError
│
├── CloudStorageError
│   ├── UploadError
│   ├── DownloadError
│   ├── BucketNotFoundError
│   └── InsufficientPermissionsError
│
├── CloudTrainingError
│   ├── PodCreationTimeoutError
│   ├── InstanceLaunchError
│   ├── SpotInterruptionError
│   ├── TrainingScriptError
│   └── BudgetExceededError
│
├── CloudPredictionError
│   ├── LambdaTimeoutError
│   ├── ServerlessInvocationError
│   ├── SchemaMismatchError
│   └── PredictionFailedError
│
└── StateManagementError
    ├── InvalidTransitionError
    ├── StateCorruptionError
    └── TransactionRollbackError
```

### Error Handling Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     ERROR HANDLING FLOW                         │
└─────────────────────────────────────────────────────────────────┘

   Operation
       │
       ▼
   Try Execute
       │
       ├─────────────────────┐
       │                     │
   Success               Exception
       │                     │
       ▼                     ▼
   Return Result    ┌─────────────────┐
                    │ Classify Error  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
          Transient      Permanent      Critical
              │              │              │
              ▼              ▼              ▼
       ┌──────────┐   ┌──────────┐  ┌───────────┐
       │  Retry   │   │  Notify  │  │  Escalate │
       │  Logic   │   │   User   │  │   Admin   │
       └────┬─────┘   └────┬─────┘  └─────┬─────┘
            │              │              │
      ┌─────┴────┐         │              │
      │          │         │              │
   Success   Max Retries   │              │
      │          │         │              │
      ▼          ▼         ▼              ▼
   Return    Fallback   Return         System
   Result    Provider   Error          Alert
                │        Message
                ▼
         Alternative
         Execution
```

### Error Recovery Strategies

| Error Type | Strategy | Recovery Actions |
|------------|----------|------------------|
| **PodCreationTimeout** | Retry with fallback | 1. Retry same GPU (max 2 attempts)<br>2. Try different GPU type<br>3. Fallback to AWS<br>4. Fallback to local |
| **SpotInterruption** | Checkpoint & restart | 1. Save partial progress to storage<br>2. Create new instance<br>3. Resume from checkpoint |
| **BudgetExceeded** | Stop with notification | 1. Terminate instance immediately<br>2. Save partial results<br>3. Notify user with cost breakdown |
| **SchemaMismatch** | User correction | 1. Show missing/extra columns<br>2. Offer manual feature selection<br>3. Suggest retrain with new schema |
| **LambdaTimeout** | Switch to async | 1. Retry with async invocation<br>2. Split into smaller batches<br>3. Use RunPod serverless instead |

---

## 7. Recovery Strategies

### Checkpoint-Based Recovery

```
┌─────────────────────────────────────────────────────────────────┐
│                   CHECKPOINT RECOVERY FLOW                      │
└─────────────────────────────────────────────────────────────────┘

Training Instance                    Storage                   Bot
       │                                │                       │
   [Training]                           │                       │
       │──Save Checkpoint (Epoch 25)───>│                       │
       │                                │                       │
   [Spot Interruption Detected]         │                       │
       │──Save Final Checkpoint────────>│                       │
       │  (Epoch 50, partial)            │                       │
       │                                │                       │
   [Instance Terminated]                │                       │
       │                                │                       │
       │                                │<──Detect Interruption─│
       │                                │                       │
   [New Instance Launched]              │                       │
       │<──Download Checkpoint──────────│                       │
       │   (Epoch 50)                   │                       │
       │                                │                       │
   [Resume Training from Epoch 50]      │                       │
       │                                │                       │
   [Training Continues...]              │                       │
       │──Save Checkpoint (Epoch 75)───>│                       │
       │                                │                       │
   [Training Complete]                  │                       │
       │──Save Final Model─────────────>│                       │
       │                                │                       │
```

### Multi-Layer Fallback

```
Primary: RunPod GPU Pod
    │
    ├─ Health Check Failed
    │  │
    │  └─> Fallback to AWS EC2 Spot Instance
    │          │
    │          ├─ Spot Capacity Unavailable
    │          │  │
    │          │  └─> Fallback to AWS EC2 On-Demand
    │          │          │
    │          │          ├─ Budget Exceeded
    │          │          │  │
    │          │          │  └─> Fallback to Local Training
    │          │          │          │
    │          │          │          └─ Execute on Bot Server
    │          │          │
    │          │          └─ Success → Return Results
    │          │
    │          └─ Success → Return Results
    │
    └─ Success → Return Results
```

### State Recovery After Bot Restart

```python
def restore_interrupted_workflows():
    """
    Restore all active workflows on bot restart.

    Steps:
    1. Query database for non-complete states
    2. For each active workflow:
       a. Load state snapshot
       b. Check if cloud resources still running
       c. Resume monitoring OR clean up
    3. Notify users of resumed workflows
    """
    active_workflows = db.query("""
        SELECT user_id, conversation_id, state, state_data
        FROM workflow_states
        WHERE state NOT IN ('COMPLETE', 'ERROR_STATE')
        AND expires_at > NOW()
    """)

    for workflow in active_workflows:
        user_id = workflow['user_id']
        state = workflow['state']
        data = workflow['state_data']

        if state == 'MONITORING_TRAINING':
            # Resume monitoring
            job_id = data.get('job_id')
            if job_id:
                resume_training_monitor(user_id, job_id)

        elif state in ['LAUNCHING_TRAINING', 'AWAITING_S3_DATASET']:
            # Clean up partial resources
            cleanup_partial_resources(user_id, data)
            transition_to_error(user_id, "Bot restarted during workflow")

        # Notify user
        send_telegram_message(
            user_id,
            f"Resumed workflow from {state}. Use /status to check progress."
        )
```

---

## Architecture Best Practices

### Design Principles

1. **Separation of Concerns**
   - Clear boundaries between providers
   - Abstract interfaces for all cloud operations
   - State management isolated from business logic

2. **Fail-Safe Defaults**
   - Always fallback to local on cloud failure
   - Auto-terminate expensive resources
   - Never leave user in undefined state

3. **Idempotency**
   - All cloud operations can be safely retried
   - Storage operations use unique keys
   - Training jobs tagged with idempotency tokens

4. **Cost Control**
   - Hard budget limits enforced
   - Real-time cost tracking
   - Automatic resource termination

5. **User Experience**
   - Transparent error messages
   - Progress updates every 30 seconds
   - Clear cost estimates before confirmation

### Scalability Considerations

- **Horizontal Scaling:** Bot instances share state via database
- **Rate Limiting:** Provider API calls throttled (100/min RunPod, 5000/s AWS)
- **Connection Pooling:** Reuse HTTP connections to cloud providers
- **Caching:** Model metadata cached in Redis (5-minute TTL)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-08
**Related:** DEPLOYMENT_GUIDE.md, API_REFERENCE.md, CLOUD_TRAINING_GUIDE.md
