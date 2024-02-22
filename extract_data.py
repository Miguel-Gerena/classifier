import tarfile
import os

directory = './data/'

for filename in os.listdir(directory):
    if filename.endswith('.tar.gz') and filename != '2011.tar.gz':
        filepath = os.path.join(directory, filename)
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path=directory)
        os.remove(filepath)  # This line deletes the .tar.gz file after extraction
