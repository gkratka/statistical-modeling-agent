#!/bin/bash
#
# AWS Lambda Deployment Packaging Script
#
# This script packages the Lambda prediction handler and its dependencies
# into a deployment-ready ZIP file for AWS Lambda.
#
# Usage:
#   ./scripts/cloud/package_lambda.sh
#
# Output:
#   dist/lambda_deployment.zip - Ready for Lambda upload
#
# Requirements:
#   - Python 3.9+ with pip
#   - Write access to project directory
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Directories
LAMBDA_DIR="$PROJECT_ROOT/lambda"
DIST_DIR="$PROJECT_ROOT/dist"
PACKAGE_DIR="$DIST_DIR/package"
OUTPUT_ZIP="$DIST_DIR/lambda_deployment.zip"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}AWS Lambda Deployment Packaging${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Step 1: Validate Lambda directory exists
echo -e "${YELLOW}Step 1: Validating Lambda directory...${NC}"
if [ ! -d "$LAMBDA_DIR" ]; then
    echo -e "${RED}Error: Lambda directory not found at $LAMBDA_DIR${NC}"
    exit 1
fi

if [ ! -f "$LAMBDA_DIR/prediction_handler.py" ]; then
    echo -e "${RED}Error: prediction_handler.py not found in $LAMBDA_DIR${NC}"
    exit 1
fi

if [ ! -f "$LAMBDA_DIR/requirements.txt" ]; then
    echo -e "${RED}Error: requirements.txt not found in $LAMBDA_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Lambda directory validated${NC}"
echo ""

# Step 2: Clean previous build artifacts
echo -e "${YELLOW}Step 2: Cleaning previous build artifacts...${NC}"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"
mkdir -p "$PACKAGE_DIR"
echo -e "${GREEN}✓ Build directory cleaned${NC}"
echo ""

# Step 3: Install dependencies to package directory
echo -e "${YELLOW}Step 3: Installing Python dependencies...${NC}"
echo "This may take a few minutes..."

# Use pip to install dependencies to package directory
python3 -m pip install \
    --target "$PACKAGE_DIR" \
    --requirement "$LAMBDA_DIR/requirements.txt" \
    --upgrade \
    --quiet

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to install dependencies${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dependencies installed to package directory${NC}"
echo ""

# Step 4: Copy Lambda handler to package directory
echo -e "${YELLOW}Step 4: Copying Lambda handler...${NC}"
cp "$LAMBDA_DIR/prediction_handler.py" "$PACKAGE_DIR/"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to copy prediction_handler.py${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Lambda handler copied${NC}"
echo ""

# Step 5: Create deployment ZIP
echo -e "${YELLOW}Step 5: Creating deployment ZIP...${NC}"

# Navigate to package directory to create ZIP with correct structure
cd "$PACKAGE_DIR"

# Create ZIP with all contents
zip -r9 "$OUTPUT_ZIP" . -q

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to create deployment ZIP${NC}"
    exit 1
fi

cd "$PROJECT_ROOT"

echo -e "${GREEN}✓ Deployment ZIP created${NC}"
echo ""

# Step 6: Display package information
echo -e "${YELLOW}Step 6: Package information${NC}"
ZIP_SIZE=$(du -h "$OUTPUT_ZIP" | cut -f1)
FILE_COUNT=$(unzip -l "$OUTPUT_ZIP" | tail -1 | awk '{print $2}')

echo "  Location: $OUTPUT_ZIP"
echo "  Size: $ZIP_SIZE"
echo "  Files: $FILE_COUNT"
echo ""

# Step 7: Validate package contents
echo -e "${YELLOW}Step 7: Validating package contents...${NC}"

# Check for required files
REQUIRED_FILES=(
    "prediction_handler.py"
    "pandas/__init__.py"
    "sklearn/__init__.py"
    "joblib/__init__.py"
    "numpy/__init__.py"
    "boto3/__init__.py"
)

ALL_VALID=true
for file in "${REQUIRED_FILES[@]}"; do
    if unzip -l "$OUTPUT_ZIP" | grep -q "$file"; then
        echo -e "${GREEN}  ✓ $file${NC}"
    else
        echo -e "${RED}  ✗ $file - MISSING${NC}"
        ALL_VALID=false
    fi
done

echo ""

if [ "$ALL_VALID" = false ]; then
    echo -e "${RED}Error: Package validation failed - missing required files${NC}"
    exit 1
fi

# Step 8: Display deployment instructions
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Packaging Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Deployment Instructions:${NC}"
echo ""
echo "1. AWS Console:"
echo "   - Navigate to AWS Lambda console"
echo "   - Create/update your function"
echo "   - Upload: $OUTPUT_ZIP"
echo "   - Set handler: prediction_handler.lambda_handler"
echo "   - Set runtime: Python 3.9+"
echo "   - Set memory: 512MB+ (adjust based on model size)"
echo "   - Set timeout: 30s+ (adjust based on data size)"
echo ""
echo "2. AWS CLI:"
echo "   aws lambda update-function-code \\"
echo "     --function-name your-function-name \\"
echo "     --zip-file fileb://$OUTPUT_ZIP"
echo ""
echo "3. Terraform/CloudFormation:"
echo "   Use the ZIP file as your deployment package source"
echo ""
echo -e "${GREEN}Ready for deployment!${NC}"
