from __future__ import print_function
import warnings
warnings.filterwarnings('ignore','.*conversion.*')

import os
import h5py
import zipfile
import shutil
import requests
import numpy as np
from PIL import Image
import argparse

################################
# Dataset with Google Drive IDs#
################################

dataset = {
    'CUHK03': '1BO4G9gbOTJgtYIB0VNyHQpZb8Lcn-05m',
    'Market1501': '0B2FnquNgAXonU3RTcE1jQlZ3X0E',
    'Market1501Attribute' : '1YMgni5oz-RPkyKHzOKnYRR2H3IRKdsHO',
    'DukeMTMC': '1qtFGJQ6eFu66Tt7WG85KBxtACSE8RBZ0',
    'DukeMTMCAttribute' : '1eilPJFnk_EHECKj2glU_ZLLO7eR3JIiO',
    'MSMT17':'18EFJN4gfgv18ayL01S7EUm-kSvQvyNmE',
    'NTUCampus' : '1UFobPpi6xP0LTzo3aheE605CZ3XNWo9L',
}

##########################
# Google Drive Downloader#
##########################

def gdrive_downloader(destination, id):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

###########################
# ReID Dataset Downloader#
###########################

def PersonReID_Dataset_Downloader(save_dir, dataset_name):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_dir_exist = os.path.join(save_dir , dataset_name)

    if not os.path.exists(save_dir_exist):
        temp_dir = os.path.join(save_dir , 'temp')

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        destination = os.path.join(temp_dir , dataset_name)

        id = dataset[dataset_name]

        print("Downloading %s" % dataset_name)
        gdrive_downloader(destination, id)

        zip_ref = zipfile.ZipFile(destination)
        print("Extracting %s" % dataset_name)
        zip_ref.extractall(save_dir)
        zip_ref.close()
        shutil.rmtree(temp_dir)
        print("Done")
    else:
        print("Dataset Check Success: %s exists!" %dataset_name)

#For United Testing and External Use
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Name and Dataset Directory')
    parser.add_argument(dest="save_dir", action="store", default="~/Datasets/",help="")
    parser.add_argument(dest="dataset_name", action="store", type=str,help="")
    args = parser.parse_args() 
    PersonReID_Dataset_Downloader(args.save_dir,args.dataset_name)
