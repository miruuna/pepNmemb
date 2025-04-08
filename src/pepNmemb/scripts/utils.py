from __future__ import annotations

from typing import Any, Dict, Optional


class NoneValueError(Exception):
    """Exception raised when a required value is None."""

    def __init__(self, key: str, message: Optional[str] = None):
        """
        Init error with message
        """
        self.key = key
        self.message = message or f"Value for key '{key}' cannot be None"
        super().__init__(self.message)


def validate_dict_values(data: Dict[str, Any], raise_error: bool = True) -> Dict[str, str]:
    """
    Validate that no values in the dictionary are None.

    Args:
        data: Dictionary to validate
        raise_error: Whether to raise an error (True) or just return invalid keys (False)

    Returns:
        Dictionary containing only invalid keys and their values

    Raises:
        NoneValueError: If any value is None and raise_error is True
    """
    none_values = {k: v for k, v in data.items() if v is None}

    if none_values and raise_error:
        # Raise error for the first None value found
        first_key = next(iter(none_values))
        raise NoneValueError(first_key)

    return none_values


def process_file(filename: str) -> Dict[str, str]:
    """
    Process the file provided as argument.
    Extracts key-value pairs in the format key=value where each pair is on a separate line.
    Returns a dictionary of the extracted key-value pairs.
    """
    key_value_dict = {}
    try:
        with open(filename) as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line or line.startswith("#"):  # Skip empty lines and comments
                    continue

                if "=" in line:
                    key, value = line.split("=", 1)  # Split on the first '=' only
                    key = key.strip()
                    value = value.strip()
                    if value == "":
                        value = None

                    if key in key_value_dict:
                        print(
                            f"Warning: Duplicate key '{key}' found on line {line_number}, \
                                value will be overwritten"
                        )

                    key_value_dict[key] = value
                else:
                    print(
                        f"Warning: Line {line_number} does not contain a \
                              key-value pair in the format 'key=value', skipping: '{line}'"
                    )

            print(
                f"Successfully processed file '{filename}'. \
                    Found {len(key_value_dict)} key-value pairs."
            )
            return key_value_dict

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return {}
    except Exception as e:
        print(f"Error processing file: {e}")
        return {}
