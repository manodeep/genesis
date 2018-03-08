#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from optparse import OptionParser
import sys
import h5py

sys.path.append('../')
import forest_sorter as fs
 
def parse_inputs():
    """

    Parses the command line input arguments.
    
    Parameters
    ----------

    None.   
 
    Returns
    ----------

    opt: optparse.Values.  Required.  
        Values from the OptionParser package.  
        Values are accessed through ``opt.Value`` and cast into a dictionary using ``vars(opt)`` 
    """

    parser = OptionParser()

    parser.add_option("-f", "--fname_in", dest="fname_in", 
                      help="Path to test HDF5 data. Default: ./test_data.hdf5", 
                      default="./test_data.hdf5")
    parser.add_option("-o", "--fname_out", dest="fname_out", 
                      help="Path to sorted output HDF5 data file. Default: ./test_sorted.hdf5",
                      default="./test_sorted.hdf5")
    parser.add_option("-n", "--NHalos_test", dest="NHalos_test", 
                      help="Minimum number of halos to test using. Default: 10,000", 
                      default = 10000, type = int) 
    parser.add_option("-s", "--sort_id", dest="sort_id", 
                      help="Field name for the key we are sorting on. Default: ForestID.",  
                      default = "ForestID")
    parser.add_option("-m", "--mass_def", dest="sort_mass", 
                      help="Field name for the mass key we are sorting on. Default: Mass_200mean.", 
                      default = "Mass_200mean")
    parser.add_option("-i", "--HaloID", dest="halo_id", 
                      help="Field name for halo ID. Default: ID.", 
                      default = "ID")
    parser.add_option("-p", "--ID_fields", dest="ID_fields", 
                      help="Field names for those that contain non-merger IDs.  Default: ('ID').",
                      default = ("ID")) 
    parser.add_option("-x", "--index_mult_factor", dest="index_mult_factor", 
                      help="Conversion factor to go from a unique, per-snapshot halo index to a \
                      temporally unique haloID.  Default: 1e12.", 
                      default = 1e12)

    (opt, args) = parser.parse_args()

    # Print some useful startup info. #
    print("")
    print("Running test functions")
    print("Performing tests on a minimum of {0} halos.".format(opt.NHalos_test))
    print("The HaloID field for each halo is '{0}'.".format(opt.halo_id)) 
    print("Sorting on the '{0}' field.".format(opt.sort_id))
    print("Sub-Sorting on the '{0}' field.".format(opt.sort_mass))
    print("")

    return opt

def test_sorted_indices(halo_id, match_id, match_mass, indices, opt):
    """

    Checks the indices of the sorted array to ensure that the sorting has been performed properly. 

    Parameters
    ----------

    halo_id: HDF5 dataset.  Required. 
        The HDF5 dataset containing the IDs of the halos.
        Note: This is the ID of the halo, not the index of its position.
    
    match_id, match_mass: HDF5 dataset.  Required.
        The HDF5 dataset containing the fields that the halos were sorted by.  The first dataset is the 'outer-sort' with the second dataset specifying the 'inner-sort'.
        For Treefrog the default options here are the fields "ForestID" and "Mass_200mean".
        See "sort_dataset" for specific example with Treefrog.

    indices: array-like of indices.  Required.
        Array containing the indices sorted using the specified keys.

    opt: Dictionary.  Required
        Dictionary containing the option parameters specified at runtime.  Used to print some debug messages.

    Returns
    ----------

    True if test passes, False otherwise. 

    """
    
    for idx in range(len(indices) -1):        
        if (match_id[indices[idx]] > match_id[indices[idx + 1]]): # First check to ensure that the sort was done in ascending order.

            print("For Halo ID {0} we had a {4} of {1}.  After sorting via lexsort with key {4}, the next Halo has ID {2} and {4} of {3}.".format(halo_id[indices[idx]], match_id[indices[idx]], halo_id[indices[idx + 1]], match_id[indices[idx + 1]], opt["sort_id"]))
            print("Since we are sorting using {0} they MUST be in ascending order.".format(opt["sort_id"]))
            return False 

        if (match_id[indices[idx]] == match_id[indices[idx + 1]]): # Then for the inner-sort, check that the sort within each ID group was done correctly.
            if (match_mass[indices[idx]] > match_mass[indices[idx]]):

                print("For Halo ID {0} we had a {4} of {1}.  After sorting via lexsort with key {4}, the next Halo has ID {2} and {4} of {3}.".format(halo_id[indices[idx]], match_id[indices[idx]], halo_id[indices[idx + 1]], match_id[indices[idx + 1]], opt["sort_id"]))
                print("However we have sub-sorted within {0} with the key {1}.  Halo ID {2} has a {1} value of {3} whereas the sequentially next halo with ID {4} has a {1} value of {5}".format(opt["sort_id"], opt["sort_mass"], halo_id[indices[idx]], match_mass[indices[idx]], halo_id[indices[idx + 1]], mass[indices[idx + 1]])) 
                print("Since we are sub-sorting using {0} they MUST be in ascending order.".format(opt["sort_mass"]))
                return False
 
    return True 

def test_check_haloIDs(opt):
    """

    As all haloIDs are temporally unique, given the snapshot number we should be able to recreate the haloIDs. 
    This test checks the passed haloIDs and snapshot number to ensure the haloIDs match the given formula.
 
    Parameters
    ----------

    file_haloIDs: array-like of integers. Required.
        Array containing the haloIDs from the HDF5 file at the snapshot of interest. 
    SnapNum: integer. Required
        Snapshot that the halos are located at. 
 
    Returns
    ----------

    True if test passes, False otherwise.     

    """
   
    files = [opt["fname_in"], opt["fname_out"]]

    for file_to_test in files: 
        with h5py.File(file_to_test, "r") as f_in:
            Snap_Keys, Snap_Nums = fs.get_snapkeys_and_nums(f_in.keys())
        
            for snap_key in Snap_Keys:
                if len(f_in[snap_key][opt["halo_id"]]) == 0:  # Skip empty snapshots. 
                    continue

                file_haloIDs = f_in[snap_key][opt["halo_id"]][:] 
                generated_haloIDs = fs.index_to_temporalID(np.arange(len(file_haloIDs)), 
                                                                     Snap_Nums[snap_key],
                                                                     opt["index_mult_factor"])
    
        if (np.array_equal(generated_haloIDs, file_haloIDs) == False):
            print("The HaloIDs within file '{0}' were not correct.".format(file_to_test))
            print("HaloIDs were {0} and the expected IDs were {1}.".format(file_haloIDs,
                                                                           generated_haloIDs))
            print("If this is the test input data file, then your input data is wrong!\n\
If this is the test sorted output file, contact jseiler@swin.edu.au.")
            return False

    return True

def test_output_file(filename_in, filename_out, opt): 
    """

    Ensures that the output data file for a specified snapshot has been written in the properly sorted order.

    Note: The merger ID fields are not validated because they are wrong by design.  If HaloID 1900000000001 had a descendant pointer (i.e., a 'Head' point in Treefrog) of 2100000000003, this may not be true because the ID of Halo 2100000000003 may be changed. 
 
    Parameters
    ----------

    snapshot_group_in, snapshot_group_out: HDF5 group.  Required.
        Groups for the specified snapshot for the input/output HDF5 data files.

    SnapNum: integer.  Required.
        Snapshot number we are doing the comparison for.

    indices: array-like of integers.  Required.
        Indices that map the input data to the sorted output data.  
        These indices were created by sorting on ``opt["sort_id"]`` (outer-sort) and ``opt["sort_mass"]`` (inner-sort).  See function ``get_sort_indices()``.
 
    opt: Dictionary.  Required.
        Dictionary containing the option parameters specified at runtime.  Used to specify the field names we are sorting on. 
     
    Returns
    ----------

    True if test passes, False otherwise.     
    """

    if status == False:
        return False

    status = test_sorted_indices(snapshot_group_out[opt["halo_id"]], snapshot_group_out[opt["sort_id"]], snapshot_group_out[opt["sort_mass"]], np.arange(0, len(snapshot_group_out[opt["halo_id"]])), opt) # Test that the sorting using the ``sort_id`` ('ForestID' in Treefrog) and ``sort_mass`` ('Mass_200mean' in Treefrog) were performed properly. If they're sorted properly, the indices that "sort" them should correspond to a simply np.arange.
    if status == False:
        return False
   
    for field in snapshot_group_out.keys(): # Now let's check each field.
        if (field in opt["ID_fields"]) == True: # If the field is an ID field ignore it (see docstring). 
            continue
        input_data = snapshot_group_in[field][:] # Grab all the data we are going to check.
        input_data_sorted = input_data[indices]
        output_data = snapshot_group_out[field][:]
        if (np.array_equal(output_data, input_data_sorted) == False): # Output data must be equal to the sorted input data.
            print("For snapshot number {0}, there was a mismatch for field {1} for the sorted input data and the output data\nThe raw input data is {2}, with correpsonding sorted order {3}, which does match the output data of {4}".format(SnapNum, field, input_data, input_data_sorted, output_data))
            return False
    
    return True

def create_test_input_data(opt):

    with h5py.File(opt["fname_in"], "r") as f_in, h5py.File("./my_test_data.hdf5", "w") as f_out:
        NHalos = 0
        
        Snap_Keys = [key for key in f_in.keys() if (("SNAP" in key.upper()) == True)] 
        Snap_Nums = dict() 
        for key in Snap_Keys: 
            Snap_Nums[key] = fs.snap_key_to_snapnum(key) 

        for snap_key in Snap_Keys:
            if len(f_in[snap_key][opt["halo_id"]]) == 0:  # Skip empty snapshots. 
                continue
        
            fs.copy_group(f_in, f_out, snap_key, opt)
            NHalos += len(f_in[snap_key][opt["halo_id"]])

            if NHalos >= opt["NHalos_test"]:
                break

    return "./my_test_data.hdf5"

def tests(opt):

    if "-f" in sys.argv: # User specified their own input data.
        print("You have supplied your own test input data.")
        print("Saving a small file with the first {0} Halos.".format(opt["NHalos_test"]))
        opt["fname_in"] = create_test_input_data(opt)

    fs.sort_and_write_file(opt)

    test_check_haloIDs(opt) # Test that the new temporally unique haloIDs were done properly.
    #test_output_file(opt)
 
if __name__ == '__main__':
    
    opt = parse_inputs()
    tests(vars(opt))
