"""Unit tests for schema parser with all 3 supported formats."""

import pytest
from src.utils.schema_parser import SchemaParser, ParsedSchema
from src.utils.exceptions import ValidationError


class TestSchemaParserJSON:
    """Test JSON format parsing."""

    def test_valid_json_list_features(self):
        """Test valid JSON with features as list."""
        input_str = '{"target": "price", "features": ["sqft", "bedrooms", "bathrooms"]}'
        result = SchemaParser.parse(input_str)

        assert result.target == "price"
        assert result.features == ["sqft", "bedrooms", "bathrooms"]
        assert result.format_detected == "json"

    def test_valid_json_string_features_not_supported(self):
        """Test JSON with features as string is not supported (must be list)."""
        input_str = '{"target": "price", "features": "sqft, bedrooms, bathrooms"}'

        with pytest.raises(ValidationError, match="'features' must be a list"):
            SchemaParser.parse(input_str)

    def test_json_missing_target(self):
        """Test JSON missing target field."""
        input_str = '{"features": ["sqft", "bedrooms"]}'

        with pytest.raises(ValidationError, match="missing 'target' field"):
            SchemaParser.parse(input_str)

    def test_json_missing_features(self):
        """Test JSON missing features field."""
        input_str = '{"target": "price"}'

        with pytest.raises(ValidationError, match="missing 'features' field"):
            SchemaParser.parse(input_str)

    def test_json_empty_target(self):
        """Test JSON with empty target."""
        input_str = '{"target": "", "features": ["sqft"]}'

        with pytest.raises(ValidationError, match="'target' cannot be empty"):
            SchemaParser.parse(input_str)

    def test_json_empty_features(self):
        """Test JSON with empty features list."""
        input_str = '{"target": "price", "features": []}'

        with pytest.raises(ValidationError, match="missing 'features' field"):
            SchemaParser.parse(input_str)

    def test_json_invalid_syntax(self):
        """Test malformed JSON."""
        input_str = '{"target": "price", "features": ["sqft"'  # Missing closing bracket

        with pytest.raises(ValidationError, match="Could not parse schema"):
            SchemaParser.parse(input_str)

    def test_json_whitespace_handling(self):
        """Test JSON with extra whitespace in values."""
        input_str = '{"target": "  price  ", "features": ["  sqft  ", "bedrooms"]}'
        result = SchemaParser.parse(input_str)

        assert result.target == "price"
        assert result.features == ["sqft", "bedrooms"]


class TestSchemaParserKeyValue:
    """Test key-value format parsing."""

    def test_valid_key_value_newlines(self):
        """Test valid key-value format with newlines."""
        input_str = """target: price
features: sqft, bedrooms, bathrooms"""
        result = SchemaParser.parse(input_str)

        assert result.target == "price"
        assert result.features == ["sqft", "bedrooms", "bathrooms"]
        assert result.format_detected == "key_value"

    def test_valid_key_value_semicolons(self):
        """Test valid key-value format with semicolons."""
        input_str = "target: price; features: sqft, bedrooms"
        result = SchemaParser.parse(input_str)

        assert result.target == "price"
        assert result.features == ["sqft", "bedrooms"]
        assert result.format_detected == "key_value"

    def test_key_value_case_insensitive(self):
        """Test key-value format is case-insensitive for keys."""
        input_str = """TARGET: price
FEATURES: sqft, bedrooms"""
        result = SchemaParser.parse(input_str)

        assert result.target == "price"
        assert result.features == ["sqft", "bedrooms"]

    def test_key_value_missing_target(self):
        """Test key-value missing target."""
        input_str = "features: sqft, bedrooms"

        with pytest.raises(ValidationError, match="missing 'target"):
            SchemaParser.parse(input_str)

    def test_key_value_missing_features(self):
        """Test key-value missing features."""
        input_str = "target: price"

        with pytest.raises(ValidationError, match="missing 'features"):
            SchemaParser.parse(input_str)

    def test_key_value_whitespace_handling(self):
        """Test key-value with extra whitespace."""
        input_str = """target:   price
features:  sqft  ,  bedrooms  , bathrooms   """
        result = SchemaParser.parse(input_str)

        assert result.target == "price"
        assert result.features == ["sqft", "bedrooms", "bathrooms"]


class TestSchemaParserSimpleList:
    """Test simple comma-separated list format parsing."""

    def test_valid_simple_list(self):
        """Test valid simple list format."""
        input_str = "price, sqft, bedrooms, bathrooms"
        result = SchemaParser.parse(input_str)

        assert result.target == "price"
        assert result.features == ["sqft", "bedrooms", "bathrooms"]
        assert result.format_detected == "simple_list"

    def test_simple_list_minimum_columns(self):
        """Test simple list with minimum 2 columns."""
        input_str = "price, sqft"
        result = SchemaParser.parse(input_str)

        assert result.target == "price"
        assert result.features == ["sqft"]

    def test_simple_list_too_few_columns(self):
        """Test simple list with only 1 column."""
        input_str = "price"

        with pytest.raises(ValidationError, match="at least 2 columns"):
            SchemaParser.parse(input_str)

    def test_simple_list_whitespace_handling(self):
        """Test simple list with extra whitespace."""
        input_str = "  price  ,  sqft  ,  bedrooms  "
        result = SchemaParser.parse(input_str)

        assert result.target == "price"
        assert result.features == ["sqft", "bedrooms"]


class TestSchemaParserValidation:
    """Test schema validation rules."""

    def test_duplicate_columns(self):
        """Test detection of duplicate column names."""
        input_str = "price, sqft, bedrooms, sqft"  # Duplicate 'sqft'

        with pytest.raises(ValidationError, match="Duplicate column names"):
            SchemaParser.parse(input_str)

    def test_duplicate_case_insensitive(self):
        """Test duplicate detection is case-insensitive."""
        input_str = "price, Sqft, bedrooms, SQFT"

        with pytest.raises(ValidationError, match="Duplicate column names"):
            SchemaParser.parse(input_str)

    def test_target_in_features(self):
        """Test target cannot be in features (detected as duplicate)."""
        input_str = '{"target": "price", "features": ["sqft", "price", "bedrooms"]}'

        with pytest.raises(ValidationError, match="Duplicate column names detected"):
            SchemaParser.parse(input_str)

    def test_empty_string_input(self):
        """Test empty input raises error."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            SchemaParser.parse("")

    def test_whitespace_only_input(self):
        """Test whitespace-only input raises error."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            SchemaParser.parse("   \n\t   ")


class TestSchemaParserFormatDetection:
    """Test format auto-detection priority."""

    def test_json_takes_priority(self):
        """Test JSON format is tried first."""
        # This is valid JSON with single feature element containing comma
        input_str = '{"target": "price", "features": ["sqft, bedrooms"]}'
        result = SchemaParser.parse(input_str)

        assert result.format_detected == "json"
        # Features is a list with one element (not split by comma)
        assert result.features == ["sqft, bedrooms"]

    def test_key_value_over_simple_list(self):
        """Test key-value format is tried before simple list."""
        # This could be parsed as simple list, but key-value takes priority
        input_str = "target: price\nfeatures: sqft, bedrooms"
        result = SchemaParser.parse(input_str)

        assert result.format_detected == "key_value"

    def test_fallback_to_simple_list(self):
        """Test fallback to simple list when others fail."""
        # Not JSON, not key-value, must be simple list
        input_str = "price, sqft, bedrooms"
        result = SchemaParser.parse(input_str)

        assert result.format_detected == "simple_list"


class TestSchemaParserDisplay:
    """Test schema display formatting."""

    def test_format_schema_short_features(self):
        """Test display format with few features."""
        schema = ParsedSchema(
            target="price",
            features=["sqft", "bedrooms", "bathrooms"],
            raw_input="",
            format_detected="json"
        )

        display = SchemaParser.format_schema_for_display(schema)

        assert "**Target:** `price`" in display
        assert "`sqft`" in display
        assert "`bedrooms`" in display
        assert "`bathrooms`" in display
        assert "**Format:** json" in display

    def test_format_schema_many_features(self):
        """Test display format truncates long feature lists."""
        features = [f"feature_{i}" for i in range(10)]
        schema = ParsedSchema(
            target="target",
            features=features,
            raw_input="",
            format_detected="simple_list"
        )

        display = SchemaParser.format_schema_for_display(schema)

        # Should show first 5 and indicate more
        assert "`feature_0`" in display
        assert "`feature_4`" in display
        assert "(+5 more)" in display
        assert "`feature_9`" not in display  # Should be truncated


class TestSchemaParserEdgeCases:
    """Test edge cases and special characters."""

    def test_column_names_with_underscores(self):
        """Test column names with underscores."""
        input_str = "house_price, square_feet, num_bedrooms"
        result = SchemaParser.parse(input_str)

        assert result.target == "house_price"
        assert "square_feet" in result.features
        assert "num_bedrooms" in result.features

    def test_column_names_with_numbers(self):
        """Test column names with numbers."""
        input_str = "target1, feature2, feature3"
        result = SchemaParser.parse(input_str)

        assert result.target == "target1"
        assert "feature2" in result.features

    def test_unicode_column_names(self):
        """Test column names with unicode characters."""
        input_str = "价格, 面积, 卧室"  # price, area, bedrooms in Chinese
        result = SchemaParser.parse(input_str)

        assert result.target == "价格"
        assert "面积" in result.features

    def test_mixed_format_attempts(self):
        """Test input that doesn't clearly match any format."""
        # This is ambiguous - neither JSON, key-value, nor proper simple list
        input_str = "target price features sqft"

        with pytest.raises(ValidationError, match="Could not parse schema"):
            SchemaParser.parse(input_str)


class TestSchemaParserRealWorldExamples:
    """Test real-world usage examples."""

    def test_housing_dataset_json(self):
        """Test housing dataset schema via JSON."""
        input_str = '''
        {
            "target": "median_house_value",
            "features": [
                "longitude", "latitude", "housing_median_age",
                "total_rooms", "total_bedrooms", "population",
                "households", "median_income"
            ]
        }
        '''
        result = SchemaParser.parse(input_str)

        assert result.target == "median_house_value"
        assert len(result.features) == 8
        assert "median_income" in result.features

    def test_housing_dataset_key_value(self):
        """Test housing dataset schema via key-value."""
        input_str = """
        target: median_house_value
        features: longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income
        """
        result = SchemaParser.parse(input_str)

        assert result.target == "median_house_value"
        assert len(result.features) == 8

    def test_housing_dataset_simple_list(self):
        """Test housing dataset schema via simple list."""
        input_str = "median_house_value, longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income"
        result = SchemaParser.parse(input_str)

        assert result.target == "median_house_value"
        assert len(result.features) == 8

    def test_titanic_dataset(self):
        """Test Titanic dataset schema."""
        input_str = '{"target": "Survived", "features": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]}'
        result = SchemaParser.parse(input_str)

        assert result.target == "Survived"
        assert "Pclass" in result.features
        assert "Age" in result.features
