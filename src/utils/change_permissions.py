import os
import stat
#from tqdm import tqdm

def change_permissions(folder_path):
    for root, dirs, files in os.walk(folder_path):
        if '.venv' in dirs:
            #dirs.remove('.venv')
            #dirs.remove('.ipynb_checkpoints')
            pass
        for file in files:
            file_path = os.path.join(root, file)
            os.chmod(file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

if __name__ == "__main__":
    folder_path = "/compass/Shared/Users/bogdanov/ml_tokamak/imgs"
    change_permissions(folder_path)
    print("Permissions changed for all files in the folder.")