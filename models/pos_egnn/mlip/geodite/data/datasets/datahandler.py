from abc import abstractmethod
from os import makedirs
from os.path import exists, join
from typing import Any, Dict, Generator, List, Optional

import wget


class TemplateDataHandler:
    # download files of a dataset and return those files in standardized format
    def __init__(
        self,
        root_folder: str,
        license: str,
        download_link: str,
        n_entries: Optional[int | None],
    ):
        self.root_folder = root_folder
        self.lincense = license
        self.download_link = download_link
        self.n_entries = n_entries

    def get_data_stream(self, n_files_to_skip: int = 0) -> Generator[Dict[str, Any], None, None]:
        if not exists(self.root_folder):
            makedirs(self.root_folder)
        file_path = self._download()
        gen = self._stream_data(file_path, n_files_to_skip)

        return gen

    @abstractmethod
    def _stream_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Method to read a compressed file and return a list with its contents

        Args:
            -file_path: The file location

        Returns:
            -file_components: A list containing dictionaries, containing the necessary information inside.
                             "file_contents" and "id" are always required to be inside each dictionary.
                             Additional fields may be passed inside the dictionary as kwargs to the parser.
        """
        pass

    def _download(self) -> str:
        # Download the file if it does not exist
        filename = self.name
        file_path = join(self.root_folder, filename)
        if not exists(file_path):
            print("Downloading...")
            wget.download(self.download_link, out=file_path, bar=self.bar_in_MB)
            print()
        return file_path

    @staticmethod
    def bar_in_MB(current, total, width=80):
        return wget.bar_adaptive(round(current / 1024 / 1024, 2), round(total / 1024 / 1024, 2), width) + " MB"

    @property
    def name(self):
        return type(self).__name__
