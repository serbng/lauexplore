import numpy as np
import pandas as pd

from LaueTools.CrystalParameters import Prepare_Grain
from LaueTools.lauecore import SimulateLaue_full_np
from LaueTools.dict_LaueTools import dict_Materials, dict_CCD

def simulate(material, orientation, calibration_parameters, **kwargs):
    """Simulate a Laue pattern

    Args:
        material                (str): Material name. Must be a key of the materials dictionary inside LaueTools.
                                       If not, the kwarg 'material_dictionary' is expected to be passed with the
                                       lattice parameters of the material.
        orientation      (np.ndarray): 3 by 3 matrix representing the orientation of the crystal in a reference
                                       system of the BM32 experimental hutch. That is, where
                                           * u_x : direction of the X-ray beam
                                           * u_z : direction towards the detector
                                           * u_y : - u_x cross u_z
        calibration_parameters (list): List of the five calibration parameters:
                                           *  detector distance: distance between the detector and the sample
                                           * x_center, y_center: position that we encounter on the detector if, starting from the sample, 
                                                                 we trace a vertical line. Takes into account in-plane shifts of the camera.
                                           *    x_beta, x_gamma: deviation angles of the stabilizing stage from the flat position. Takes
                                                                 into account the shift of the relative angle between the sample and the beam.

    Returns:
        result (pandas.DataFrame): dataframe containing the result. Has columns (h,k,l,2θ,χ,X,Y,Energy).
                                       (h,k,l) : Miller indices of the reflection
                                       (2θ,χ)  : Scattering angles
                                       (X,Y)   : Position of the reflection on the detector
                                       Energy  : Energy of the reflection in keV
        
    Kwargs:
        Emin               (float): Minimum energy of the polychromatic X-ray beam in keV. Default value 5
        Emax               (float): Maximum energy of the polychromatic X-ray beam in keV. Default value 5
        material_dictionary (dict): Dictionary containing information of materials lattice parameters.
                                        Structure:
                                            * key    (str): material name
                                            * value (list): ["material", [a,b,c,α,β,γ], "extintion_rule"]
                                        Default value:
                                            Dictionary of materials inside the LaueTools library.
        camera_label         (str): Must be in the LaueTools dictionary containing the detector parameters.
                                    Defaults to "sCMOS", that is the camera currently in the beamline as of
                                    July 2025.
        detector_diameter  (float): Defaults to 148 [mm].
    """
    # Simulation parameters
    Emin = kwargs.pop("Emin",  5)
    Emax = kwargs.pop("Emax", 25)
    material_dictionary = kwargs.pop("material_dictionary", dict_Materials)
    simulation_parameters = Prepare_Grain(material, orientation, dictmaterials=material_dictionary)
    
    # Camera-related parameters
    camera_label      = kwargs.pop("camera_label", "sCMOS")
    detector_diameter = kwargs.pop("detector_diameter", 148.1212) * 1.75 # Detector diameter is not present in
    pixel_size        = dict_CCD[camera_label][1]                        # the dict_CCD["sCMOS"] for some reason
    frame_shape       = dict_CCD[camera_label][0]
    
    # Actual simulation
    result = SimulateLaue_full_np(simulation_parameters, Emin, Emax, calibration_parameters, # mandatory positional arguments
                                  detectordiameter = detector_diameter,
                                  pixelsize        = pixel_size,
                                  dim              = frame_shape,
                                  dictmaterials    = material_dictionary,
                                  kf_direction     = 'Z>0', # default value
                                  removeharmonics  = 0)     # default value
    # Convert from tuple to list
    result = list(result)

    # Some simulated reflection fall outside of the detector, remove them
    x, y = result[3], result[4]
    to_keep  = (x > 0) & (x < frame_shape[1])
    to_keep &= (y > 0) & (y < frame_shape[0])
    
    # Keep only the peaks that fall within the detector and round them
    for i in range(len(result)):
        result[i] = result[i][to_keep]
        result[i] = np.round(result[i], decimals=3)
    
    # Preparing dataframe
    simulation_result = {
        "h": result[2][:,0],
        "k": result[2][:,1],
        "l": result[2][:,2],
        "2θ": result[0],
        "χ": result[1],
        "X": result[3],
        "Y": result[4],
        "Energy": result[5]
    }
    
    return pd.DataFrame.from_dict(simulation_result).convert_dtypes()
