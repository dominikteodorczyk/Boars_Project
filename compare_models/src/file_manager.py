import os


class FileManager:
    def __init__(self, output_dir: str, logger):
        self.output_dir = output_dir
        self.logger = logger

    def create_output_dir_for_file(self, file_name: str):
        output_dir = os.path.join(self.output_dir, file_name)
        if not os.path.exists(output_dir):
            self.logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)
        return output_dir
