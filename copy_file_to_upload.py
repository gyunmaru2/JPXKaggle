# %%
import os
import shutil

wd = "/workspaces/JPXKaggle/"
files = ['fin_etl.py','preparefile.py','tech_etl.py',
    'create_dataset_kaggle.py'
]

for file in files :
    if os.path.exists(wd+"upload_py/"+file) :
        os.remove(wd+"upload_py/"+file)
    shutil.copyfile(wd+file,wd+"upload_py/"+file)

# %%
