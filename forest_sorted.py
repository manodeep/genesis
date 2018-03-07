#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import h5py
from tqdm import tqdm
from optparse import OptionParser

def get_sorted_indices(dataset, snap_key, opt):
    """

    Sorts the input HDF5 dataset using 2-keys given in "opt".  The first key specifies the order of the "outer-sort" with the second key specifying the order of the "inner-sort" within each group sorted by the first key. 

    Example:
        Outer-sort uses ForestID and inner-sort used Mass_200mean.
        ForestID = [1, 4, 39, 1, 1, 4]
        Mass_200mean = [4e9, 10e10, 8e8, 7e9, 3e11, 5e6]
        
        Then the indices would be [0, 3, 4, 5, 1, 2]

    If the debug option has been specified in the runtime options (opt["debug"] == 1), then a check is run to ensure that the sorting has been correctly done.

    Parameters
    ----------

    dataset: HDF5 dataset.  Required.
        Input HDF5 dataset that we are sorting over. The data structure is assumed to be HDF5_File -> List of Snapshot Numbers -> Halo properties/pointers.

    snap_key: String.  Required.
        The field name for the snapshot we are accessing.

    opt: Dictionary.  Required.
        Dictionary containing the option parameters specified at runtime.  Used to specify the field names we are sorting on. 
        
    Returns
    ----------

    indices: numpy-array.  Required.
        Array containing the indices that sorts the keys for the specified dataset. 

    """ 
    indices = np.lexsort((dataset[snap_key][opt["sort_mass"]], dataset[snap_key][opt["sort_id"]])) # Sorts via the ID then sub-sorts via the mass

    if opt["debug"] == 1: 
        test_sorted_indices(f[snap_key][halo_id], f[snap_key][sort_id], f[snap_key][sort_mass], indices, opt)

    return indices

def test_sorted_indices(halo_id, match_id, match_mass, indices, opt):
    """

    Checks the indices of the sorted array to ensure that the sorting has been performed properly. 
    Only called in the debug option is specified.

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

    None.  Function will raise a RunTimeError if the indices have not been sorted properly. 

    """
    
    for idx in range(len(indices) -1):        
        if (match_id[indices[idx]] > match_id[indices[idx + 1]]): # First check to ensure that the sort was done in ascending order.

            print("For Halo ID {0} we had a {4} of {1}.  After sorting via lexsort with key {4}, the next Halo has ID {2} and {4} of {3}.".format(halo_id[indices[idx]], match_id[indices[idx]], halo_id[indices[idx + 1]], match_id[indices[idx + 1]], opt["sort_id"]))
            print("Since we are sorting using {0} they MUST be in ascending order.".format(opt["sort_id"]))
            raise RuntimeError 

        if (match_id[indices[idx]] == match_id[indices[idx + 1]]): # Then for the inner-sort, check that the sort within each ID group was done correctly.
            if (match_mass[indices[idx]] > match_mass[indices[idx]]):

                print("For Halo ID {0} we had a {4} of {1}.  After sorting via lexsort with key {4}, the next Halo has ID {2} and {4} of {3}.".format(halo_id[indices[idx]], match_id[indices[idx]], halo_id[indices[idx + 1]], match_id[indices[idx + 1]], opt["sort_id"]))
                print("However we have sub-sorted within {0} with the key {1}.  Halo ID {2} has a {1} value of {3} whereas the sequentially next halo with ID {4} has a {1} value of {5}".format(opt["sort_id"], opt["sort_mass"], halo_id[indices[idx]], match_mass[indices[idx]], halo_id[indices[idx + 1]], mass[indices[idx + 1]])) 
                print("Since we are sub-sorting using {0} they MUST be in ascending order.".format(opt["sort_mass"]))
                raise RuntimeError                 

def parse_snap_field(Snap_Field):
    """

    Given the name of a snapshot field, we wish to find the snapshot number associated with this field.
    This is necessary because the 0th snapshot field may not be snapshot 000 and there could be missing snapshots (snapshot 39 is followed by snapshot 40).
    The fields passed to this function are usually those which contain the word "Snap" or "snap".

    Note: This logic of handling snapshot fields will fail if the snapshot fields do not include the word 'Snap' or 'snap', and if they contain numbers other than the snapshot number (sNAp_1_032 for example).
    
    Parameters
    ----------

    Snap_Field: String.  Required. 
        The name of the snapshot field.    
    
    Returns
    ----------

    SnapNum: integer.  Required.
        The snapshot number that corresponds to the snapshot field. 

    """

    SnapNum = ""

    for letter in Snap_Field: # Go through each letter within the snapshot field,
        if (letter.isdigit() == True): # When a number is found,
            SnapNum = "{0}{1}".format(SnapNum, letter) # Concatenate that number with the others.

    return int(SnapNum) # Recast as integer before returning.


def parse_inputs():
    """

    Parses the command line input arguments.  If there has not been an input or output file name specified a RuntimeError will be raised. 
    
    Parameters
    ----------

    None.   
 
    Returns
    ----------

    opt: optparse.Values.  Required.  
        Values from the OptionParser package.  Values are accessed through ``opt.Value`` and cast into a dictionary using ``vars(opt)`` 
    """

    parser = OptionParser()

    parser.add_option("-f", "--fname_in", dest="fname_in", help="Path to the input HDF5 data file. Required.")
    parser.add_option("-o", "--fname_out", dest="fname_out", help="Path to the output HDF5 data file. Required.")
    parser.add_option("-s", "--sort_id", dest="sort_id", help="Field name for the key we are sorting on. Default: ForestID.", default = "ForestID")
    parser.add_option("-m", "--mass_def", dest="sort_mass", help="Field name for the mass key we are sorting on. Default: Mass_200mean.", default = "Mass_200mean")
    parser.add_option("-i", "--HaloID", dest="halo_id", help="Field name for halo ID. Default: ID.", default = "ID")
    parser.add_option("-d", "--debug", dest="debug", help="Set to 1 to toggle debug mode. Default: 0 (off).", default = 0)
    parser.add_option("-p", "--ID_fields", dest="ID_fields", help="Field names for those that contain IDs.  Default: ('ID', 'Tail', 'Head', 'NextSubHalo', 'Dummy1', 'Dumm2').", default = ("ID", "Tail", "Head", "NextSubHalo", "Dummy", "Dummy")) 

    (opt, args) = parser.parse_args()

    if (opt.fname_in == None or opt.fname_out == None): # If the required parameters have not been supplied, throw an exception.
        parser.print_help()
        raise RuntimeError

    # Print some useful startup info. #
    print("")
    print("The HaloID field for each halo is '{0}'.".format(opt.halo_id)) 
    print("Sorting on the '{0}' field.".format(opt.sort_id))
    print("Sub-Sorting on the '{0}' field.".format(opt.sort_mass))
    print("")

    return opt

def ID_to_temporalID(index, SnapNum):
    """

    Given a haloID local to a snapshot with number ``SnapNum``, this function returns the ID that accounts for the snapshot number. 
    
    Parameters
    ----------

    index: array-like of integers, or integer. Required.
        Array or single value that describes the snapshot-local haloID.
    SnapNum: integer. Required
        Snapshot that the halo/s are/is located at.
 
    Returns
    ----------

    index: array-like of integers, or integer. Required.    
        Array or single value that contains the temporally unique haloID.         
    """

    temporalID = SnapNum*int(1e12) + index + 1

    return temporalID

def test_check_haloIDs(file_haloIDs, SnapNum):
    """

    As all haloIDs are temporally unique, given the snapshot number we should be able to recreate the haloIDs. 
    If this is not the case a ValueError is raised.   
 
    Parameters
    ----------

    file_haloIDs: array-like of integers. Required.
        Array containing the haloIDs from the HDF5 file at the snapshot of interest. 
    SnapNum: integer. Required
        Snapshot that the halos are located at. 
 
    Returns
    ----------

    None.  If the haloIDs within the HDF5 file do not match the expected values a ValueError is raised. 

    """

    generated_haloIDs = ID_to_temporalID(np.arange(len(file_haloIDs)), SnapNum)
    
    if (np.array_equal(generated_haloIDs, file_haloIDs) == False):
        raise ValueError("The HaloIDs for snapshot {0} did not match the formula.\nHaloIDs were {1} and the expected IDs were {2}".format(SnapNum, file_haloIDs, generated_haloIDs))
    
def copy_field(file_in, file_out, key, field, opt):
    """

    Copies the field (and it's nested data-structure) within a HDF5 group into a new HDF5 file with the same data-structure.
 
    Parameters
    ----------

    file_in, file_out: Open HDF5 files.  Required.
        HDF5 files for the data being copied (file_in) and the file the data is being copied to (file_out). 
    key, field: Strings.  Required.
        Name of the HDF5 group/dataset being copied. 
    opt: Dictionary.  Required.
        Dictionary containing the option parameters specified at runtime.  Used to specify the field names we are sorting on. 
     
    Returns
    ----------

    None. 

    """

    ## TODO: Add a check that both the input and output files are open and HDF5 files.

    group_path = file_in[key][field].parent.name
    group_id = file_out.require_group(group_path)
    name = "{0}/{1}".format(key, field)
    f_in.copy(name, group_id, name = field)

    if opt['debug'] == 1:
        print("Created field {0} for snap_key {1}".format(field, snap_key))

def test_output_file(snapshot_group_in, snapshot_group_out, SnapNum, indices, opt): 
    """

    Ensures that the output data file for a specified snapshot has been written in the properly sorted order.
    If this is not the case a ValueError is raised.

    Note: The ID fields are not validated because they are wrong by design.  If HaloID 1900000000001 had a descendant pointer (i.e., a 'Head' point in Treefrog) of 2100000000003, this may not be true because the ID of Halo 2100000000003 may be changed. 
 
    Parameters
    ----------

    snapshot_group_in, snapshot_group_out: HDF5 group.  Required.
        Groups for the specified snapshot for the input/output HDF5 data files.
    SnapNum: integer.  Required.
        Snapshot number we are doing the comparison for.
    indices: array-like of integers.  Required.
        Indices that map the input data to the sorted output data.  
        These indices were created by sorting on ``opt["sort_id"]`` (outer-sort) and ``opt["sort_mass"]`` (inner-sort).  See function ``get_sorted_indices()``. 
    opt: Dictionary.  Required.
        Dictionary containing the option parameters specified at runtime.  Used to specify the field names we are sorting on. 
     
    Returns
    ----------

    None. If the test fails a ValueError is raised. 

    """

    test_check_haloIDs(snapshot_group_out[opt["halo_id"]], SnapNum) # Test that the new temporally unique haloIDs were done properly.
    test_sorted_indices(snapshot_group_out[opt["halo_id"]], snapshot_group_out[opt["sort_id"]], snapshot_group_out[opt["sort_mass"]], np.arange(0, len(snapshot_group_out[opt["halo_id"]])), opt) # Test that the sorting using the ``sort_id`` ('ForestID' in Treefrog) and ``sort_mass`` ('Mass_200mean' in Treefrog) were performed properly. If they're sorted properly, the indices that "sort" them should correspond to a simply np.arange.
   
    for field in snapshot_group_out.keys(): # Now let's check each field.
        if (field in opt["ID_fields"]) == True: # If the field is an ID field ignore it (see docstring). 
            continue
        input_data = snapshot_group_in[field][:] # Grab all the data we are going to check.
        input_data_sorted = input_data[indices]
        output_data = snapshot_group_out[field][:]
        if (np.array_equal(output_data, input_data_sorted) == False): # Output data must be equal to the sorted input data.
            raise ValueError("For snapshot number {0}, there was a mismatch for field {1} for the sorted input data and the output data\nThe raw input data is {2}, with correpsonding sorted order {3}, which does match the output data of {4}".format(SnapNum, field, input_data, input_data_sorted, output_data))
     
if __name__ == '__main__':

    opt = parse_inputs()
       
    outfile = "/Users/100921091/Desktop/Genesis/my_test.hdf5"
    with h5py.File(opt.fname_in, "r") as f_in,  h5py.File(opt.fname_out, "w") as f_out:

        oldID_to_newID = {} # Dictionary to go from the oldID to the newID.
     
        snap_keys = [] 
        snap_nums = {} 
        for field in list(f_in.keys()): # Want to generate a list of snapshot fields and the associate snapshot number.            
            if (field.find("Snap") > -1 or field.find("snap") > -1  or field.find("SNAP") > -1): # .find returns -1 if the string is not found.
                snap_keys.append(field) # Remember the name of the field. 
                snap_nums[field] = parse_snap_field(field) # Find out what snapshot number the name corresponds to. 
        
        ID_maps = {}
        created_dict = 0
        snapshot_indices = {}

        print("")
        print("Generating the dictionary to map the oldIDs to the newIDs.") 
        for snap_key in tqdm(snap_keys):
            if (len(f_in[snap_key][opt.halo_id]) == 0): # If there aren't any halos at this snapshot, move along.
                continue 

            if opt.debug:
                test_check_haloIDs(f_in[snap_key][opt.halo_id], snap_nums[snap_key])

            indices = get_sorted_indices(f_in, snap_key, vars(opt)) # Get the indices that will correctly sort the snapshot dataset by ForestID and halo mass.
            old_haloIDs = f_in[snap_key][opt.halo_id][:][indices] # Grab the HaloIDs in the sorted order.
            new_haloIDs = ID_to_temporalID(np.arange(len(indices)), snap_nums[snap_key]) # Generate new IDs.                       
            oldIDs_to_newIDs = dict(zip(old_haloIDs, new_haloIDs)) # Create a dictionary mapping between the old and new IDs.
            
            snapshot_indices[snap_key] = indices # Move the indices a dictionary keyed by the snapshot field.
            
            ## We need a dictionary that contains ID mappings for every snapshot (for merger pointers). ##
            ## If this is the first snapshot with halos, create the dictionary, otherwise append it to the global one. ##
            if created_dict == 0:
                ID_maps = oldIDs_to_newIDs
                created_dict = 1                                
            else: 
                ID_maps = {**ID_maps, **oldIDs_to_newIDs} # Taken from https://stackoverflow.com/questions/8930915/append-dictionary-to-a-dictionary 
        print("Done!")
        print("")

        ## At this point we have the dictionaries that map the oldIDs to the newIDs in addition to the indices that control the sorting of the array. ##
        ## At this point we can loop through all the fields within each halo halo within each snapshot and if the field is 'ID', 'Tail', etc, generate the new field using the oldID->newID map. ##
        ## While going through each field, we will then write out the data into a new HDF5 file in the order specified by indices; write(read_file[field][indices]). ## 

        print("")
        print("Now writing out the snapshots in the sorted order.")
        for count, key in (enumerate(f_in.keys())): # Loop through snapshots.            
            for field in f_in[key]: # Loop through each field 

                copy_field(f_in, f_out, key, field, vars(opt)) # Copy the field into the output file.
                if (key in snap_keys) == False: # We only need to adjust re-writing etc for the snapshot fields.
                    continue 

                for idx in range(len(f_in[key][opt.halo_id])): # Loop through each halo within the field.
                    if (field in opt.ID_fields) == True: # If we need to update the ID for this field.                        
                        oldID = f_in[key][field][idx] 
                        f_out[key][field][idx] = ID_maps[oldID] # Overwrite the oldID with the newID.

                if len(f_in[key][opt.halo_id]) > 0:
                    f_out[key][field][:] = f_out[key][field][:][snapshot_indices[key]] # Then reorder the properties to be in the sorted order.
   
            if (key in snap_keys) == True and len(f_in[key][opt.halo_id]) > 0 and opt.debug: # If there were halos for this snapshot and we want to debug it.
                test_output_file(f_in[key], f_out[key], snap_nums[key], snapshot_indices[key], vars(opt)) # Do a final check to ensure the properties were written out properly. 
 
            if (count > 20 and opt.debug):
                break
        print("Done!")        
        print("")        
