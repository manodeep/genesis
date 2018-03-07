#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import h5py
from tqdm import tqdm
from optparse import OptionParser

LHalo_Desc_full = [
('Descendant',          np.int32),
('FirstProgenitor',     np.int32),
('NextProgenitor',      np.int32),
('FirstHaloInFOFgroup', np.int32),
('NextHaloInFOFgroup',  np.int32),
('Len',                 np.int32),
('M_mean200',           np.float32),
('Mvir',                np.float32),
('M_TopHat',            np.float32),
('Pos',                 (np.float32, 3)),
('Vel',                 (np.float32, 3)),
('VelDisp',             np.float32),
('Vmax',                np.float32),
('Spin',                (np.float32, 3)),
('MostBoundID',         np.int64),
('SnapNum',             np.int32),
('Filenr',              np.int32),
('SubHaloIndex',        np.int32),
('SubHalfMass',         np.float32)
                 ]

names = [LHalo_Desc_full[i][0] for i in range(len(LHalo_Desc_full))]
formats = [LHalo_Desc_full[i][1] for i in range(len(LHalo_Desc_full))]
LHalo_Desc = np.dtype({'names':names, 'formats':formats}, align=True)

id_mult_factor = 1e12
NumSnaps = 200

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

    dataset: HDF5 dataset, required.
        Input HDF5 dataset that we are sorting over. The data structure is assumed to be HDF5_File -> List of Snapshot Numbers -> Halo properties/pointers.

    snap_key: String, required.
        The field name for the snapshot we are accessing.

    opt: Dictionary, required.
        Dictionary containing the option parameters specified at runtime.  Used to specify the field names with are sorting on. 
        
    Returns
    ----------

    indices: numpy-array, required.
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

    halo_id: HDF5 dataset, required. 
        The HDF5 dataset containing the IDs of the halos.
        Note: This is the ID of the halo, not the index of its position.
    
    match_id, match_mass: HDF5 dataset, required.
        The HDF5 dataset containing the fields that the halos were sorted by.  The first dataset is the 'outer-sort' with the second dataset specifying the 'inner-sort'.
        For Treefrog the default options here are the fields "ForestID" and "Mass_200mean".
        See "sort_dataset" for specific example with Treefrog.

    indices: array-like, required.
        Array containing the indices sorted using the specified keys.

    opt: Dictionary, required
        Dictionary containing the option parameters specified at runtime.  Used to print some debug messages.

    Returns
    ----------

    status: integer
        1 if the indices have been sorted properly, 0 otherwise.

    """
    
    for idx in range(len(indices) -1):        
        if (match_id[indices[idx]] > match_id[indices[idx + 1]]): # First check to ensure that the sort was done in ascending order.

            print("For Halo ID {0} we had a {4} of {1}.  After sorting via lexsort with key {4}, the next Halo has ID {2} and {4} of {3}.".format(halo_id[indices[idx]], match_id[indices[idx]], halo_id[indices[idx + 1]], match_id[indices[idx + 1]], opt["sort_id"]))
            print("Since we are sorting using {0} they MUST be in ascending order.".format(opt["sort_id"]))
            #raise Blah 

        if (match_id[indices[idx]] == match_id[indices[idx + 1]]): # Then for the inner-sort, check that the sort within each ID group was done correctly.
            if (match_mass[indices[idx]] > match_mass[indices[idx]]):

                print("For Halo ID {0} we had a {4} of {1}.  After sorting via lexsort with key {4}, the next Halo has ID {2} and {4} of {3}.".format(halo_id[indices[idx]], match_id[indices[idx]], halo_id[indices[idx + 1]], match_id[indices[idx + 1]], opt["sort_id"]))
                print("However we have sub-sorted within {0} with the key {1}.  Halo ID {2} has a {1} value of {3} whereas the sequentially next halo with ID {4} has a {1} value of {5}".format(opt["sort_id"], opt["sort_mass"], halo_id[indices[idx]], match_mass[indices[idx]], halo_id[indices[idx + 1]], mass[indices[idx + 1]])) 
                print("Since we are sub-sorting using {0} they MUST be in ascending order.".format(opt["sort_mass"]))
                #raise Blah 


def parse_snap_field(Snap_Field):

    SnapNum = ""

    for letter in Snap_Field:
        if (letter.isdigit() == True): 
            SnapNum = "{0}{1}".format(SnapNum, letter)

    return int(SnapNum)


def parse_inputs():

    parser = OptionParser()

    parser.add_option("-f", "--fname_in", dest="fname_in", help="Path to the input HDF5 data file. Required.")
    parser.add_option("-o", "--fname_out", dest="fname_out", help="Path to the output HDF5 data file. Required.")
    parser.add_option("-s", "--sort_id", dest="sort_id", help="Field name for the key we are sorting on. Default: ForestID.", default = "ForestID")
    parser.add_option("-m", "--mass_def", dest="sort_mass", help="Field name for the mass key we are sorting on. Default: Mass_200mean.", default = "Mass_200mean")
    parser.add_option("-i", "--HaloID", dest="halo_id", help="Field name for halo ID. Default: ID.", default = "ID")
    parser.add_option("-d", "--debug", dest="debug", help="Set to 1 to toggle debug mode. Default: 0 (off).", default = 0)
    parser.add_option("-p", "--ID_fields", dest="ID_fields", help="Field names for those that contain IDs.  Default: ('ID', 'Tail', 'Head', 'NextSubHalo', 'Dummy1', 'Dumm2').", default = ("ID", "Tail", "Head", "NextSubHalo", "Dummy", "Dummy")) 

    (opt, args) = parser.parse_args()

    if (opt.fname_in == None or opt.fname_out == None):
        parser.print_help()
        exit()

    print("The HaloID field for each halo is {0}.".format(opt.halo_id)) 
    print("Sorting on the {0} field.".format(opt.sort_id))
    print("Sub-Sorting on the {0} field.".format(opt.sort_mass))

    return (opt, args)

def ID_to_temporalID(index, SnapNum):

    temporalID = SnapNum*int(1e12) + index + 1

    return temporalID

def test_check_haloIDs(snap_file, SnapNum, opt):

    targetIDs = ID_to_temporalID(np.arange(len(snap_file[opt['halo_id']])), SnapNum)
    haloIDs =snap_file[opt['halo_id']]
    
    if (np.array_equal(targetIDs, haloIDs) == False):
        raise ValueError("The HaloIDs did not match the formula!")
    
def copy_field(file_in, file_out, snap_key, field, opt):

    group_path = file_in[snap_key][field].parent.name
    group_id = file_out.require_group(group_path)
    name = "{0}/{1}".format(snap_key, field)
    f_in.copy(name, group_id, name = field)

    if opt['debug'] == 1:
        print("Created field {0} for snap_key {1}".format(field, snap_key))
 
if __name__ == '__main__':

    (opt, args) = parse_inputs()

    outfile = "/Users/100921091/Desktop/Genesis/my_test.hdf5"
    with h5py.File(opt.fname_in, "r") as f_in,  h5py.File(opt.fname_out, "w") as f_out:

        oldID_to_newID = {} # Dictionary to go from the oldID to the newID.
     
        snap_fields = [] 
        snap_nums = {} 
        for field in list(f_in.keys()): # Want to generate a list of snapshot fields and the associate snapshot number.            
            if (field.find("Snap") > -1 or field.find("snap") > -1): # .find returns -1 if the string is not found.
                snap_fields.append(field) # Remember the name of the field. 
                snap_nums[field] = parse_snap_field(field) # Find out what snapshot number the name corresponds to. 
        
        ID_maps = {}
        snapshot_indices = {}

        print("")
        print("Generating the dictionary to map the oldIDs to the newIDs.") 
        for snap_key in tqdm(snap_fields):
            if (len(f_in[snap_key][opt.halo_id]) == 0): # If there aren't any halos at this snapshot, move along.
                continue 

            if opt.debug:
                test_check_haloIDs(f_in[snap_key], snap_nums[snap_key], vars(opt))

            indices = get_sorted_indices(f_in, snap_key, vars(opt)) # Get the indices that will correctly sort the snapshot dataset by ForestID and halo mass.
            old_haloIDs = f_in[snap_key][opt.halo_id][:][indices] # Grab the HaloIDs in the sorted order.
            new_haloIDs = ID_to_temporalID(np.arange(len(indices)), snap_nums[snap_key]) # Generate new IDs.                       
            oldIDs_to_newIDs = dict(zip(old_haloIDs, new_haloIDs)) # Create a dictionary mapping between the old and new IDs.
            
            snapshot_indices[snap_key] = indices # Move the indices and the map dictionary to a dictionary keyed by the snapshot field. 
            ID_maps[snap_key] = oldIDs_to_newIDs
        print("Done!")
        print("")

        ## At this point we have the dictionaries that map the oldIDs to the newIDs in addition to the indices that control the sorting of the array. ##
        ## At this point we can loop through all the fields within each halo halo within each snapshot and if the field is 'ID', 'Tail', etc, generate the new field using the oldID->newID map. ##
        ## While going through each field, we will then write out the data into a new HDF5 file in the order specified by indices; write(read_file[field][indices]). ## 

        print("")
        print("Now writing out the snapshots in the sorted order.")

        for count, snap_key in tqdm(enumerate(snap_fields)): # Loop through snapshots.            
            for field in f_in[snap_key]: # Loop through each field 
                copy_field(f_in, f_out, snap_key, field, vars(opt)) # Copy the field into the output file. 
                for idx in range(len(f_in[snap_key][opt.halo_id])): # Loop through each halo within the field.
                    if (field in opt.ID_fields) == True: # If we need to update the ID for this field.                    
                        oldID = f_in[snap_key][opt.halo_id][idx] 
                        f_out[snap_key][field][idx] = ID_maps[snap_key][oldID] # Overwrite the oldID with the newID.  We can overwrite because we now 

                if len(f_in[snap_key][opt.halo_id]) > 0:
                    f_out[snap_key][field][:] = f_out[snap_key][field][:][snapshot_indices[snap_key]] # Then reorder the properties to be in the sorted order.
            if (count > 20 and opt.debug):
                exit()

        '''
        for key in tqdm(snap_fields): # Loop through all the fields. 
            for field in f_in[key]: # Loop through each field.
                copy_field(f_in, f_out, snap_key, field, vars(opt)) # Copy the field into the output file. 
                for idx in range(len(f_in[snap_key][opt.halo_id])): # Loop through each halo within the field.
                    if (field in opt.ID_fields) == True: # If we need to update the ID for this field.                    
                        oldID = f_in[snap_key][opt.halo_id][idx] 
                        f_out[snap_key][field][idx] = ID_maps[snap_key][oldID] # Overwrite the oldID with the newID.  We can overwrite because we now 

                if len(f_in[snap_key][opt.halo_id]) > 0:
                    f_out[snap_key][field][:] = f_out[snap_key][field][:][snapshot_indices[snap_key]] # Then reorder the properties to be in the sorted order.
        '''
        print("Done!")
        print("")        
