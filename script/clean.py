# !/usr/bin/env python
# encoding: utf-8

"""
    Clean up script for removing __pycache__, .git directories and .DS_Store files, recursively.
"""

import os, shutil


def remove_pycache():
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                dir_path = os.path.join(root, dir_name)
                shutil.rmtree(dir_path)
                print(f'\tRemoved: {dir_path}')


def remove_git():
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '.git':
                dir_path = os.path.join(root, dir_name)
                shutil.rmtree(dir_path)
                print(f'\tRemoved: {dir_path}')


def remove_ds_store():
    for root, dirs, files in os.walk('.'):
        for file_name in files:
            if file_name == '.DS_Store':
                file_path = os.path.join(root, file_name)
                os.remove(file_path)
                print(f'\tRemoved: {file_path}')


if __name__ == '__main__':
    os.chdir('..')
    print('Removing __pycache__ directories...')
    remove_pycache()

    print('Removing .git directories...')
    remove_git()

    print('Removing .DS_Store files...')
    remove_ds_store()
