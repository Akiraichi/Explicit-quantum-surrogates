"""
Custom error classes for the Aoba project.

This module defines custom exception classes that provide more informative
error messages for specific error conditions in the project.
"""


class ConfigurationError(Exception):
    """
    Exception raised for errors in the configuration of experiments.
    
    This exception should be used when a configuration parameter is invalid,
    missing, or incompatible with other parameters.
    
    Attributes:
        message -- explanation of the error
    """
    
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)