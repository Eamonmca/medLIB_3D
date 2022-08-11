from attr import attrs
from  Parsing.parse_funcs import *
from Github_code.contrib_pydicom.input_output.pydicom_series import read_files as pread
import numpy as np
import nibabel
import plotly.express as px
import plotly.io as pio
import cv2
import os
from tqdm import tqdm
from Utils.utils import *
import SimpleITK as sitk




def get_scan_dict(dcm_series, patient_ID):
    '''
    '''
    patient_dcm_collection = pread(dcm_series)
    scan_dict = {}
    new_scan_dict = {}
    for scan in patient_dcm_collection:
        series_description = processString(scan.info.SeriesDescription)
        if series_description in [""] :
            series_description = "UNNAMED_SERIES"
        else:
            series_description = processString(scan.info.SeriesDescription)
            
        dicom_series = scan._datasets
        scan_dict[f"{series_description}"] = dicom_series

    for k in scan_dict:
        new_scan_dict[k] = Scan(k, scan_dict[k], patient_ID)
        
    return new_scan_dict



class Dataset(object):
    def __init__(self, paired_list):
        pbar = tqdm(paired_list)
        for pair in pbar:
            path = pair[0]
            title = processString(pair[1])
            setattr(self, title, Patient(path, patient_ID=title))
            pbar.set_description("Processing %s" % title)
        
    @property
    def patient_list(self):
        return [var for var in vars(self) if not var.startswith("_")]
    
    @property
    def num_patients(self):
        return len(vars(self))
    
    def __len__(self):
        return len(self.patient_list)
    
    def __iter__(self):
        newdict = {k: vars(self)[k] for k in [var for var in vars(self) if not var.startswith("_")]}
        return iter(newdict.values())
    

class Patient(object):
    def __init__(self, initial_data, patient_ID = None):
        
        if patient_ID is None:
            self._patient_ID ="UNNAMED_PATIENT"
        else:
            self._patient_ID = patient_ID
            
        initial_data = get_scan_dict(initial_data, patient_ID)
        for key in initial_data:
            setattr(self, key, initial_data[key])
    
    @property
    def scan_list(self):
        return [var for var in vars(self) if not var.startswith("_")]

    @property
    def patient_id(self):
        return self._patient_ID
    
    @property
    def num_scans(self):
        len(self.scan_list)
        
    def __iter__(self): 
        newdict = {k: vars(self)[k] for k in [var for var in vars(self) if not var.startswith("_")]}
        return iter(newdict.values())
    
    def __getitem__(self, key):
        return vars(self)[key]
     
    def __len__(self):
        return len(self.scan_list)
    
    def __repr__(self):
        attrs = [var for var in vars(self) if not var.startswith("_")]
        no_scans = len(attrs)
        attr_string = lambda x: ', '.join(x)
        return f"Patient Object concisting of {no_scans} scans : {attr_string(attrs)}"
    

class Scan(object):
    
    def __init__(self, title, dcm_series, patient_ID=None):
        self.patient_ID = patient_ID
        self.vol, self.pix_dim, self.affine = dicom_to_volume(dcm_series)
        self.title = title
        self.dcm_series = dcm_series
        self.patient_ID = patient_ID
        
    @property 
    def getName(self):
        return self.__class__.__name__
         
    def to_nifti(self, output_path):
        nifti_file = nibabel.Nifti1Image(self.vol, self.affine)
        try:
            os.mkdir(os.path.join(output_path,self.patient_ID))
        except FileExistsError:
            pass
        output_path = f"{output_path}/{self.patient_ID}/{self.title}.nii.gz"
        nibabel.save(nifti_file, output_path)
    
        return output_path
    
    def to_dicom(self, output_root_dir):
        volume_3D = self.vol
        no_idx = self.vol.shape[0]
        name = self.title
        
        try:
            os.mkdir(output_root_dir)
        except FileExistsError:
            pass
        
        for idx in range(no_idx):
                try:
                    os.mkdir(os.path.join(output_root_dir,self.patient_ID))
                except FileExistsError:
                    pass
                out_path = (f"{output_root_dir}/{self.patient_ID}/{name}_{idx}.dcm")
                image = volume_3D[idx,:,:]
                image = sitk.GetImageFromArray(image)
                castFilter = sitk.CastImageFilter()
                castFilter.SetOutputPixelType(sitk.sitkInt16)

                # Convert floating type image (imgSmooth) to int type (imgFiltered)
                image = castFilter.Execute(image)
                sitk.WriteImage(image,out_path)
            
        return output_path


    def to_slices(self, output_root_dir, image_format = "PNG", return_list = False):
        volume_3D = self.vol
        no_idx = self.vol.shape[0]
        name = self.title

        slice_image_list = []

        try:
            os.mkdir(output_root_dir)
        except FileExistsError:
            pass

        if image_format == "PNG":
            for idx in range(no_idx):
                try:
                    os.mkdir(os.path.join(output_root_dir,self.patient_ID))
                except FileExistsError:
                    pass
                out_path = (f"{output_root_dir}/{self.patient_ID}/{name}_{idx}.png")
                image = volume_3D[idx,:,:]
                image = np.uint8(image)
                cv2.imwrite(out_path, image)
                slice_image_list.append(out_path)

        elif image_format == "TIFF":
            for idx in range(no_idx):
                try:
                    os.mkdir(os.path.join(output_root_dir,self.patient_ID))
                except FileExistsError:
                    pass
                out_path = (f"{output_root_dir}/{self.patient_ID}/{name}_{idx}.png")
                image = volume_3D[idx,:,:]
                image = np.uint16(image)
                cv2.imwrite(out_path, image)
                slice_image_list.append(out_path)


        else : print("Unsupported Image format for slices detected. Support may be added in future if deemed appropriate.")
        if return_list:
            return slice_image_list
        

     
    def display(self, fig=False):
        
        if is_notebook:
            pio.renderers.default = 'notebook_connected'
        
        if self.vol.shape[0] == 1:
            fig = px.imshow(self.vol[0,:,:] , title=str(self.title), binary_string=True)
        else:
            fig = px.imshow(self.vol, animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"), title= str(self.title))
            fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 50
        
        if fig :
            return fig
        else:
            return fig.show()
    
    
    def __repr__(self):
        return f"Scan Object titled {self.title}, with dimensions {self.vol.shape}"

