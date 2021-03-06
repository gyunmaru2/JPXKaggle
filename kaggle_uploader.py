import glob
import os
import json
import sys
# import time
# import yaml
# import argparse
import logging
from tqdm import tqdm
import subprocess

class kaggle_up :

    def __init__(self) :

        pass

    def upload_to_kaggle(
                        self,
                        title: str, 
                        k_id: str,  
                        path: str, 
                        comments: str,
                        update:bool,
                        logger=None,
                        extension = None,
                        subtitle='not spesified', 
                        description="not spesified",
                        isPrivate = True,
                        licenses = "unknown" ,
                        keywords = [],
                        collaborators = []
                        ):
        '''
        >> upload_to_kaggle(title, k_id, path,  comments, update)
        
        Arguments
        =========
        title: the title of your dataset.
        k_id: kaggle account id.
        path: non-default string argument of the file path of the data to be uploaded.
        comments:non-default string argument of the comment or the version about your upload.
        logger: logger object if you use logging, default is None.
        extension: the file extension of model weight files, default is ".pth"
        subtitle: the subtitle of your dataset, default is empty string.
        description: dataset description, default is empty string.
        isPrivate: boolean to show wheather to make the data public, default is True.
        licenses = the licenses description, default is "unkown"; must be one of /
        ['CC0-1.0', 'CC-BY-SA-4.0', 'GPL-2.0', 'ODbL-1.0', 'CC-BY-NC-SA-4.0', 'unknown', 'DbCL-1.0', 'CC-BY-SA-3.0', 'copyright-authors', 'other', 'reddit-api', 'world-bank'] .
        keywords : the list of keywords about the dataset, default is empty list.
        collaborators: the list of dataset collaborators, default is empty list.
    '''

        assert os.path.exists(path) ,f"""
            path {path} do not exists
        """
        if path[-1] == '/':
            path = path[:-1]

        if extension is not None :
            model_list = glob.glob(path+f'/*{extension}')
        else :
            model_list = glob.glob(path+'/*')

        if len(model_list) == 0:
            raise FileExistsError('File does not exist, check the file extention is correct \
            or the file directory exist.')
        
        #JSON?????????????????????    
        data_json =  {
            "title": title,
            "id": f"{k_id}/{title}",
            "subtitle": subtitle,
            "description": description,
            "isPrivate": isPrivate,
            "licenses": [
                {
                    "name": licenses
                }
            ],
            "keywords": [],
            "collaborators": [],
            "data": [

            ]
        }
        #JSON???????????????data???????????????  
        data_list = []
        for mdl in model_list:
            mdl_nm = mdl.replace(path+'/', '')
            mdl_size = os.path.getsize(mdl) 
            data_dict = {
                "description": comments,
                "name": mdl_nm,
                "totalBytes": mdl_size,
                "columns": []
            }
            data_list.append(data_dict)
        data_json['data'] = data_list

        
        with open(path+'/dataset-metadata.json', 'w') as f:
            json.dump(data_json, f)
        
    #?????????????????????????????????????????????kaggle API????????????
        script0 = ['kaggle',  'datasets', 'create', '-p', f'{path}' ]
    #??????????????????????????????????????????kaggle API????????????
        script1 = ['kaggle',  'datasets', 'version', '-p', f'{path}' , '-m' , f'\"{comments}\"']


        if logger:    
            logger.info(data_json)
            
            if update:
                logger.info(script1)
                logger.info(subprocess.check_output(script1))
            else:
                logger.info(script0)
                logger.info(script1)
                logger.info(subprocess.check_output(script0))
                logger.info(subprocess.check_output(script1))
                
        else:
            print(data_json)
            if update:
                print(script1)
                print(subprocess.check_output(script1))
            else:
                print(script0)
                print(script1)
                print(subprocess.check_output(script0))
                print(subprocess.check_output(script1))

