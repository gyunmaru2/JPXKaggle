# %%
import os
import shutil

wd = "/workspaces/JPXKaggle/"
files = [
    'fin_etl_online.py',
    'tech_etl_online.py', 'create_dataset.py'
]

for file in files :
    if os.path.exists(wd+"upload_py3/"+file) :
        os.remove(wd+"upload_py3/"+file)
    shutil.copyfile(wd+file,wd+"upload_py3/"+file)

# %%
