import os


class FileManager:
    """
    Class to manage file operations such as creating output directories.
    """
    def __init__(self, output_dir: str, logger: any) -> None:
        """
        Initialize the FileManager with output directory and logger.
        Args:
            output_dir (str): Path to the main output directory.
            logger (any): Logger instance for logging messages.
        """
        self.output_dir = output_dir
        self.logger = logger

    def create_output_dir_for_file(self, file_name: str) -> str:
        """
        Create an output directory for a specific file if it doesn't exist.

        Args:
            file_name (str): Name of the file (without extension) to create a directory for.
        Returns:
            str: Path to the created output directory.
        """
        output_dir = os.path.join(self.output_dir, file_name)
        if not os.path.exists(output_dir):
            self.logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)
        return output_dir
