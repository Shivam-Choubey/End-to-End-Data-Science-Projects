import sys

def error_message_detail(error, error_detail: sys):
    """
    Function to extract detailed error information: filename, line number, and error message.
    """
    # exc_info() returns: type, value, traceback. We only need the traceback (exc_tb)
    _, _, exc_tb = error_detail.exc_info()

    # Extract the filename from the traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Format a string containing the script name, line number, and the actual error message
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message


class CustomException(Exception):
    """
    A custom Exception class that inherits from the base Exception class.
    """
    def __init__(self, error_message, error_detail: sys):
        # Initialize the parent Exception class
        super().__init__(error_message)
        
        # Use our helper function to generate the detailed message
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        # This ensures that when the exception is printed, it shows the detailed message
        return self.error_message
    
    