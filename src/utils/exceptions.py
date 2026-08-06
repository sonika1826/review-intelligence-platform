"""
Custom exception classes for the Review Intelligence Platform.
"""

import sys
from pathlib import Path


class ReviewIntelligenceException(Exception):
    """
    Base exception class for the application.
    """

    def __init__(self, error: Exception, error_detail: sys):
        self.error_message = self._get_error_message(error, error_detail)
        super().__init__(self.error_message)

    @staticmethod
    def _get_error_message(error: Exception, error_detail: sys) -> str:
        """
        Build a detailed error message.
        """
        _, _, exc_tb = error_detail.exc_info()

        file_name = Path(exc_tb.tb_frame.f_code.co_filename).name
        line_number = exc_tb.tb_lineno

        return (
            f"Error occurred in '{file_name}' " f"at line {line_number}: {str(error)}"
        )

    def __str__(self) -> str:
        return self.error_message
