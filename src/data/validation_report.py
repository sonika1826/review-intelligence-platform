from dataclasses import dataclass


@dataclass(slots=True)
class ValidationReport:
    is_valid: bool
    total_rows: int
    total_columns: int
    missing_columns: list[str]
    duplicate_rows: int
    missing_values: dict[str, int]
