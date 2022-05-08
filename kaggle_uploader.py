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
                        extension = '.py',
                        subtitle='', 
                        description="",
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
        model_list = glob.glob(path+f'/*{extension}')
        if len(model_list) == 0:
            raise FileExistsError('File does not exist, check the file extention is correct \
            or the file directory exist.')
        
        if path[-1] == '/':
            raise ValueError('Please remove the backslash in the end of the path')
        #JSONファイルの作成    
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
        #JSONファイルのdata部分の更新  
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
        
    #データセットを新規で作るときのkaggle APIコマンド
        script0 = ['kaggle',  'datasets', 'create', '-p', f'{path}' , '-m' , f'\"{comments}\"']
    #データセットを更新するときのkaggle APIコマンド
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

    def run(self):
    #コマンドラインの引数からパラメータのあるyamlファイルを読み込む
        # args = get_args()
        # with open(args.config_path, 'r') as f:
        #     config = yaml.safe_load(f)

        # #yamlファイルにあるパラメータを変数に代入
        # EXT = config['EXT']
        # TRAINING = config['TRAINING']
        # USE_FINETUNE = config['USE_FINETUNE']     
        # FOLDS = config['FOLDS']
        # GROUP_GAP = config['GROUP_GAP']
        # SEED = config['SEED']
        # INPUTPATH = config['INPUTPATH']
        # NUM_EPOCH = config['NUM_EPOCH']
        # BATCH_SIZE = config['BATCH_SIZE']
        # PATIANCE = config['PATIANCE']
        # LR =config['LR']
        # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(DEVICE)
        # MDL_PATH  =config['MDL_PATH']
        # MDL_NAME =config['MDL_NAME']
        # VER = config['VER']
        # THRESHOLD = config['THRESHOLD']
        # COMMENT = config['COMMENT']
        
        MDL_NAME="etls";VER="01";EXT="py"
        MDL_PATH='./upload_py'
        #logger関連の定義
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level = logging.INFO,format=format_str, filename=f'./logs/upload_log_{MDL_NAME}_{VER}_{EXT}.log')
        logger = logging.getLogger('Log')
        
        ##https://ryoz001.com/1154.html
        # コンソール画面用ハンドラー設定
        # ハンドラーのログレベルを設定する (INFO以上を出力する)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.INFO)
        # logger と コンソール用ハンドラーの関連付け
        logger.addHandler(consoleHandler)
        # logger.info(config)
        # logger.info(sys.argv)
        VER = (VER + '_' + EXT)
        model_path = f'{MDL_PATH}'
        logger.info(model_path)
        
        title = "modules"
        k_id = "takkawa"
        path = model_path
        comments = VER
        update = False
        self.upload_to_kaggle(title, k_id, path,  comments, update,logger=logger)
    
    
if __name__ == "__main__":
    
    ku = kaggle_up()

    ku.run()