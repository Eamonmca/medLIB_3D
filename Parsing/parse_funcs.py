#!/usr/bin/env python
# coding: utf-8


import os
from tqdm import tqdm
import os, sys

 
def get_sub_directories(rootdir : str) -> list:

    '''
    Simple function to return  a list of all subdirectories given a root directory path as a string. 
    Only returns subdirectories one level down and not recursively.
    '''
    
    dir_list = []
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            dir_list.append(d)
    return dir_list




def inner_sub_dir_replace(dir_list : list) -> list:
    '''
    This function takes a list of directory paths as an argument and checks if they contain an additional
    subdirectory. If an additional subdirectory is detected it will replace the origonal path list entry 
    with a path to the subdirectory.
    
    '''
    for _dir in dir_list:
        for file in os.listdir(_dir) :
            d = os.path.join(_dir, file)
            if os.path.isdir(d):
                i = dir_list.index(d[:d.rfind('/')])
                dir_list = dir_list[:i]+[d]+dir_list[i+1:]
    return dir_list

    


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout




def get_path_id_list (dir_list):
    subject_ids = []

    for _dir in dir_list:

        if "Subject" in _dir.split("/")[-1].replace(" ", "_") :
            subject_ids.append(_dir.split("/")[-1].replace(" ", "_"))
        else :
            subject_ids.append(_dir.split("/")[-2].replace(" ", "_"))
        
    path_id_pairs_list = list(zip(dir_list, subject_ids))
        
    return path_id_pairs_list




def Dicom_to_Nifti_converter(paired_list : list, output_root_dir : str) -> list :

    subject_dir_nifti_list = []

    try:
        os.mkdir(output_root_dir)
    except: pass

    for _pair in tqdm(paired_list):
        try:
            os.mkdir(os.path.join(output_root_dir, _pair[1]))
        except: pass
        try:
            dicom2nifti.convert_directory(_pair[0], 
            os.path.join(output_root_dir, _pair[1]), 
            compression=True, reorient=True)
        except: pass

    for file in os.listdir(output_root_dir):
            d = os.path.join(output_root_dir, file)
            if os.path.isdir(d):
                subject_dir_nifti_list.append(d)
    
    return subject_dir_nifti_list
    







