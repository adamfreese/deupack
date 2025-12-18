# cache.py
# Created 2025.11.18 by Adam Freese
#
# Contains method to clear deupack's cache
# Based on the delete_files_safely method of:
# https://coderivers.org/blog/python-delete-all-files-from-directory/

import os
from pathlib import Path

def clear_cache():
    path = Path(__file__).parent / 'cache/'
    confirm = input("Clear deupack's cache? [y/n] ")
    if confirm.lower() == 'y':
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as error:
                print("Error deleting {}: {}".format(file_path,error))
