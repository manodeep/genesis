#!/usr/bin/env python
from __future__ import print_function
import numpy as np

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

def test_check_haloIDs(file_haloIDs, SnapNum, index_mult_factor):
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

    generated_haloIDs = index_to_temporalID(np.arange(len(file_haloIDs)), SnapNum, index_mult_factor)
    
    if (np.array_equal(generated_haloIDs, file_haloIDs) == False):    
        print("The HaloIDs for snapshot {0} did not match the formula.\nHaloIDs were {1} and the expected IDs were {2}".format(SnapNum, file_haloIDs, generated_haloIDs))
        return False

    return True

def test_output_file(snapshot_group_in, snapshot_group_out, SnapNum, indices, opt): 
    """

    Ensures that the output data file for a specified snapshot has been written in the properly sorted order.

    Note: The ID fields are not validated because they are wrong by design.  If HaloID 1900000000001 had a descendant pointer (i.e., a 'Head' point in Treefrog) of 2100000000003, this may not be true because the ID of Halo 2100000000003 may be changed. 
 
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

    status = test_check_haloIDs(snapshot_group_out[opt["halo_id"]], SnapNum, opt["index_mult_factor"]) # Test that the new temporally unique haloIDs were done properly.
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
