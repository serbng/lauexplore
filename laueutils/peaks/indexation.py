from pathlib import Path
import numpy as np
from LaueTools.indexingSpotsSet import spotsset
#from dict_LaueTools import dict_CCD as camera_dictionary

DEFAULT_MATCHING_RATES      = [50, 60, 80]
DEFAULT_MATCHING_ANGLE_TOLS = [0.5, 0.2, 0.1]
DEFAULT_LUT_MAX_INDEX       = 4

def refinement_dict_from_kwargs(**kwargs):
    """ Build the refinement dictionary from kwargs and return eventual remaining ones"""
    refinement_dict = {
        # I use kwargs.get for the entries that are also an input of the indexing function
        'AngleTolLUT':                kwargs.pop("LUT_tolerance_angle", 0.5),   # Tolerance angle [deg]
        'nlutmax'    :                kwargs.get("nLUTmax", DEFAULT_LUT_MAX_INDEX), # Maximum miller index checked in the LUT
        'list matching tol angles':   kwargs.get("tolerance_angles", DEFAULT_MATCHING_ANGLE_TOLS),
        'central spots indices':      kwargs.pop("spotset_A", [0]),  # spots set A 
        #number of most intense spot candidate to have a recognisable distance
        'NBMAXPROBED':                kwargs.pop("spotset_B", 10),
        'MATCHINGRATE_THRESHOLD_IAL': kwargs.pop("LUT_matching_rate", 2),
        'MATCHINGRATE_ANGLE_TOL':     kwargs.pop("LUT_tolerance_angle", 0.2),
        'MinimumMatchingRate':        kwargs.pop("LUT_matching_rate", 5),
        'MinimumNumberMatches':       kwargs.pop("min_number_matches", 15),
        'UseIntensityWeights':        kwargs.pop("UseIntensityWeights", False),
        'nbSpotsToIndex':             kwargs.pop("nb_spots_max", 10000),
    }

    return refinement_dict, kwargs

def index(peaks, material, filename, dirnameout=None, **kwargs):
    """ Wrapper of LaueTools.indexingSpotsSet.spotsset.IndexSpotsSet for easier indexation.

    Parameters
    ----------

    Keyword arguments
    ----------
    
    """
    spotset = spotsset()

    # Set mandatory positional arguments for the LaueTools indexer
    if isinstance(peaks, (str, Path)):
        spotset.importdatafromfile(peaks)
        
    elif isinstance(peaks, np.ndarray):
        if calibration_dictionary not in kwargs:
            raise(ValueError, "If a list of peaks is given, the calibration dictionary must be provided")
        # I need to manually fill the parameters of the class regarding the calibration parameters

        # Copied (excluding the existance checks) from
        # LaueTools.indexingSpotsSet.spotsset.IndexSpotsSet.importdatafromfile()
        # Lines 446 - 481
        spotset.set_dict_spotprops(peaks)
        calibration_dictionary     = kwargs.pop("calibration_dictionary")
        spotset.CCDcalibdict       = calibration_dictionary
        spotset.CCDLabel           = calibration_dictionary["CCDLabel"]
        spotset.pixelsize          = calibration_dictionary["pixelsize"]
        spotset.framedim           = calibration_dictionary["framedim"]
        spotset.dim                = calibration_dictionary["framedim"]
        spotset.detectordiameter   = calibration_dictionary["detectordiameter"]
        spotset.detectorparameters = calibration_dictionary["CCDCalibParams"]
        spotset.kf_direction       = calibration_dictionary["kf_direction"]
        spotset.nbspots            = len(peaks)
        spotset.filename           = None
        spotset.updateSimulParameters()

    refinement_dict, indexation_kwargs = refinement_dict_from_kwargs(**kwargs)
    
    Emin     = kwargs.pop("Emin",  5)
    Emax     = kwargs.pop("Emax", 28)
    database = kwargs.pop("database", None)
    
    # Additional arguments
    indexation_kwargs.update(
        {
            "angletol_list":      kwargs.pop("angle_tolerances", DEFAULT_MATCHING_ANGLE_TOLS),
            "MatchingRate_List":  kwargs.pop("matching_rates",   DEFAULT_MATCHING_RATES),
            "nbGrainstoFind":     kwargs.pop("nb_grains", "max"),
            "dirnameout_fitfile": Path.cwd() if dirnameout is None else dirnameout,
            "n_LUT":              kwargs.pop("nLUTmax", DEFAULT_LUT_MAX_INDEX)
        }
    )

    spotset.IndexSpotsSet(peaks, material, Emin, Emax, refinement_dict, database, **indexation_kwargs)
    


def check_orientation(peaks, material, orientation, filename, dirnameout=None, **kwargs):
    kwargs.update(
        {"spotset_B": 0,
         "previousResults": [1, [orientation], 5, 5]}
    )

    try:
        index(peaks, material, filename, dirnameout, **kwargs)
    except:
        pass
