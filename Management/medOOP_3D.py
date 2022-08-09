from turtle import title

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




def processString(txt):
    specialChars = "!#$%^&*()" 
    for specialChar in specialChars:
        txt = txt.replace(specialChar, '')
    txt = txt.replace(" ", "_")
    return txt


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    

def create_affine(ipp, iop, ps):
    """Generate a NIFTI affine matrix from DICOM IPP and IOP attributes.

    The ipp (ImagePositionPatient) parameter should an Nx3 array, and
    the iop (ImageOrientationPatient) parameter should be Nx6, where
    N is the number of DICOM slices in the series.

    The return values are the NIFTI affine matrix and the NIFTI pixdim.
    Note the the output will use DICOM anatomical coordinates:
    x increases towards the left, y increases towards the back.
    """
    # solve Ax = b where x is slope, intecept
    n = ipp.shape[0]
    A = np.column_stack([np.arange(n), np.ones(n)])
    x, r, rank, s = np.linalg.lstsq(A, ipp, rcond=None)
    # round small values to zero
    x[(np.abs(x) < 1e-6)] = 0.0
    vec = x[0,:] # slope
    pos = x[1,:] # intercept

    # pixel spacing should be the same for all image
    spacing = np.ones(3)
    spacing[0:2] = ps[0,:]
    if np.sum(np.abs(ps - spacing[0:2])) > spacing[0]*1e-6:
        sys.stderr.write("Pixel spacing is inconsistent!\n");

    # compute slice spacing
    spacing[2] = np.round(np.sqrt(np.sum(np.square(vec))), 7)

    # get the orientation
    iop_average = np.mean(iop, axis=0)
    u = iop_average[0:3]
    u /= np.sqrt(np.sum(np.square(u)))
    v = iop_average[3:6]
    v /= np.sqrt(np.sum(np.square(v)))

    # round small values to zero
    u[(np.abs(u) < 1e-6)] = 0.0
    v[(np.abs(v) < 1e-6)] = 0.0

    # create the matrix
    mat = np.eye(4)
    mat[0:3,0] = u*spacing[0]
    mat[0:3,1] = v*spacing[1]
    mat[0:3,2] = vec
    mat[0:3,3] = pos

    # check whether slice vec is orthogonal to iop vectors
    dv = np.dot(vec, np.cross(u, v))
    qfac = np.sign(dv)
    if np.abs(qfac*dv - spacing[2]) > 1e-6:
        sys.stderr.write("Non-orthogonal volume!\n");

    # compute the nifti pixdim array
    pixdim = np.hstack([np.array(qfac), spacing])

    return mat, pixdim


def dicom_to_volume(dicom_series):
    """Convert a DICOM series into a float32 volume with orientation.

    The input should be a list of 'dataset' objects from pydicom.
    The output is a tuple (voxel_array, voxel_spacing, affine_matrix)
    """
    # Create numpy arrays for volume, pixel spacing (ps),
    # slice position (ipp or ImagePositinPatient), and
    # slice orientation (iop or ImageOrientationPatient)
    n = len(dicom_series)
    shape = (n,) + dicom_series[0].pixel_array.shape
    vol = np.empty(shape, dtype=np.float32)
    ps = np.empty((n,2), dtype=np.float64)
    ipp = np.empty((n,3), dtype=np.float64)
    iop = np.empty((n,6), dtype=np.float64)

    for i, ds in enumerate(dicom_series):
        # create a single complex-valued image from real,imag
        image = ds.pixel_array
        try:
            slope = float(ds.RescaleSlope)
        except (AttributeError, ValueError):
            slope = 1.0
        try:
            intercept = float(ds.RescaleIntercept)
        except (AttributeError, ValueError):
            intercept = 0.0
        vol[i,:,:] = image*slope + intercept
        ps[i,:] = dicom_series[i].PixelSpacing[-2:]
        ipp[i,:] = dicom_series[i].ImagePositionPatient
        iop[i,:] = dicom_series[i].ImageOrientationPatient

    # create nibabel-style affine matrix and pixdim
    # (these give DICOM LPS coords, not NIFTI RAS coords)
    affine, pixdim = create_affine(ipp, iop, ps)
    return vol, pixdim, affine



def get_scan_dict(dcm_series, paitent_ID):
    '''
    '''
    paitent_dcm_collection = pread(dcm_series)
    scan_dict = {}
    new_scan_dict = {}
    for scan in paitent_dcm_collection:
        series_description = processString(scan.info.SeriesDescription)
        if series_description in [""] :
            series_description = "UNNAMED_SERIES"
        else:
            series_description = processString(scan.info.SeriesDescription)
            
        dicom_series = scan._datasets
        scan_dict[f"{series_description}"] = dicom_series

    for k in scan_dict:
        new_scan_dict[k] = Scan(k, scan_dict[k], paitent_ID)
        
    return new_scan_dict



class Dataset(object):
    def __init__(self, paired_list):
        pbar = tqdm(paired_list)
        for pair in pbar:
            path = pair[0]
            title = processString(pair[1])
            setattr(self, title, Paitent(path, paitent_ID=title))
            pbar.set_description("Processing %s" % title)
        
    @property
    def paitent_list(self):
        return [var for var in vars(self) if not var.startswith("_")]
    
    @property
    def num_paitents(self):
        return len(vars(self))
    
    def __len__(self):
        return len(self.paitent_list)
    
    def __iter__(self):
        newdict = {k: vars(self)[k] for k in [var for var in vars(self) if not var.startswith("_")]}
        return iter(newdict.values())
    

class Paitent(object):
    def __init__(self, initial_data, paitent_ID = None):
        if paitent_ID is None:
            self._paitent_ID ="UNNAMED_PATIENT"
        else:
            self._paitent_ID = paitent_ID
        initial_data = get_scan_dict(initial_data, paitent_ID)
        for key in initial_data:
            setattr(self, key, initial_data[key])
        
    @property    
    def getName(self):
        return self.__class__.__name__
    
    @property
    def scan_list(self):
        return [var for var in vars(self) if not var.startswith("_")]
        
        # return [var for var in vars(self) if not var.startswith("_")]
    
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
        return f"Paitent Object concisting of {no_scans} scans : {attr_string(attrs)}"
    

class Scan(object):
    
    def __init__(self, title, dcm_series, paitent_ID=None):
        self.paitent_ID = paitent_ID
        self.vol, self.pix_dim, self.affine = dicom_to_volume(dcm_series)
        self.title = title
        self.dcm_series = dcm_series
        self.paitent_ID = paitent_ID
        
    @property 
    def getName(self):
        return self.__class__.__name__
         
    def write_nifti(self, output_path):
        nifti_file = nibabel.Nifti1Image(self.vol, self.affine)
        try:
            os.mkdir(os.path.join(output_path,self.paitent_ID))
        except FileExistsError:
            pass
        output_path = f"{output_path}/{self.paitent_ID}/{self.title}.nii.gz"
        nibabel.save(nifti_file, output_path)
    
        return output_path



    def Vol_to_slices(self, output_root_dir, image_format = "PNG", return_list = False):
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
                    os.mkdir(os.path.join(output_root_dir,self.paitent_ID))
                except FileExistsError:
                    pass
                out_path = (f"{output_root_dir}/{self.paitent_ID}/{name}_{idx}.png")
                image = volume_3D[idx,:,:]
                image = np.uint8(image)
                cv2.imwrite(out_path, image)
                slice_image_list.append(out_path)

        elif image_format == "TIFF":
            for idx in range(no_idx):
                try:
                    os.mkdir(os.path.join(output_root_dir,self.paitent_ID))
                except FileExistsError:
                    pass
                out_path = (f"{output_root_dir}/{self.paitent_ID}/{name}_{idx}.png")
                image = volume_3D[idx,:,:]
                image = np.uint16(image)
                cv2.imwrite(out_path, image)
                slice_image_list.append(out_path)


        else : print("Unsupported Image format for slices detected. Support may be added in future if deemed appropriate.")
        if return_list:
            return slice_image_list

    
    def display_3D_volume(self, fig=False):
        
        if is_notebook:
            pio.renderers.default = 'notebook_connected'

        img = self.vol
        fig = px.imshow(img, animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"), title= "CT_SCAN",)
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 50
        
        if fig :
            return fig
        else:
            return fig.show()
    
    
    def __repr__(self):
        return f"Scan Object titled {self.title}, with dimensions {self.vol.shape}"

