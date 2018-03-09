#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import h5py

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

def copy_halo_properties(LHalo, TreefrogHalos, treefrog_idx): 
    """
    Copies the halo properties (e.g., Mvir, position etc) from the TreefrogHalo data structure into the LHalo data structure.

    Parameters
    ----------

    LHalo: structured array with dtype 'LHalo_Desc', required
        LHalo data structure that is to be filled. This is the actual structure not the index.
    
    TreefrogHalos: HDF5 file, required
        HDF5 file that contains all halos at the current snapshot.

    treefrog_idx: integer, required
        Index (not ID) of the Treefrog Halo that we are copying the propertiesfrom.
 
    """
    
    LHalo['Len'] = TreefrogHalos['npart'][treefrog_idx]

    LHalo['M_mean200'] = TreefrogHalos['Mass_200mean'][treefrog_idx]
    LHalo['Mvir'] = TreefrogHalos['Mass_200mean'][treefrog_idx] ### NOT IN TREEFROG
    LHalo['M_TopHat'] = TreefrogHalos['Mass_200mean'][treefrog_idx] ### NOT IN TREEFROG

    LHalo['Pos'][0] = TreefrogHalos['Xc'][treefrog_idx]
    LHalo['Pos'][1] = TreefrogHalos['Yc'][treefrog_idx]
    LHalo['Pos'][2] = TreefrogHalos['Zc'][treefrog_idx]

    LHalo['Vel'][0] = TreefrogHalos['VXc'][treefrog_idx]
    LHalo['Vel'][1] = TreefrogHalos['VYc'][treefrog_idx]
    LHalo['Vel'][2] = TreefrogHalos['VZc'][treefrog_idx]

    LHalo['VelDisp'] = TreefrogHalos['sigV'][treefrog_idx]
    LHalo['Vmax'] = TreefrogHalos['Vmax'][treefrog_idx]

    LHalo['Spin'][0] = TreefrogHalos['Lx'][treefrog_idx]
    LHalo['Spin'][1] = TreefrogHalos['Ly'][treefrog_idx]
    LHalo['Spin'][2] = TreefrogHalos['Lz'][treefrog_idx]
    
    LHalo['MostBoundID'] = TreefrogHalos['npart'][treefrog_idx] ### NOT IN TREEFROG
    LHalo['SnapNum'] = int(TreefrogHalos['ID'][treefrog_idx] / id_mult_factor) # Snapshot of halo is encoded within ID, divide by the factor.
    
    return LHalo 

if __name__ == '__main__':

    fname = "/Users/100921091/Desktop/Genesis/VELOCIraptor.tree.t4.unifiedhalotree.withforestid.snap.hdf.data"

    with h5py.File(fname, "r") as f:

        #Nforests = max(f['Snap_199']['ForestID']) # Total number of forests.
        #print("There are a total of {0} forests within the data.".format(Nforests))

        Nforests = 10 # Set to a small number for testing.
        #treefrogID_to_lhaloID = []
        treefrogID_to_lhaloID = {}

        for forestID in range(1, Nforests): # Iterate for each forest. Note: ForestID starts at 1.
            NHalos = 0
            offset = 0
            for snap_idx in range(NumSnaps): 
                snap_name = 'Snap_{0:03d}'.format(snap_idx)
                treefrog_indices = np.where((f[snap_name]["ForestID"][:] == forestID))[0] # Indexes (NOT ID) in the data file for the halos in this forest.
               
                print("Snap {0}".format(snap_idx)) 
                if (len(treefrog_indices) > 0):

                    treefrog_indices = list(treefrog_indices) # Convert to list because h5py is screwy with numpy arrays.
                    treefrog_IDs = f[snap_name]["ID"][treefrog_indices] 
                    for count, ID in enumerate(treefrog_IDs):
                        treefrogID_to_lhaloID[ID] = offset + count # Update the dictionary.
                                        
                    NHalos += len(treefrog_indices) # Count how many halos for this forest there are.
                    offset += len(treefrog_indices) # Update the offset for the LHalo indexing.                                
            print("For Forest {0} there are {1} Halos".format(forestID, NHalos))

            # Now that we know how many halos there are for this forest, can allocate storage space.

            LHalos = np.full(NHalos, -1, dtype = LHalo_Desc) # Assign storage and fill with -1.
          
            for snap_idx in range(200):
                snap_name = 'Snap_{0:03d}'.format(snap_idx)            
                treefrog_indices = np.where((f[snap_name]["ForestID"][:] == forestID))[0]

                for lhalo_idx, treefrog_idx in enumerate(treefrog_indices):
                    LHalos[lhalo_idx] = copy_halo_properties(LHalos[lhalo_idx], f[snap_name], treefrog_idx) # Copy over the properties of the halo into the structure.

                    descendant_snap = int(f[snap_name]['Head'][treefrog_idx] / id_mult_factor) # Snapshot of the descendant.
                    descendant_snap_name = 'Snap_{0:03d}'.format(descendant_snap) 

                    treefrog_descendantID = f[snap_name]['Head'][treefrog_idx] # This is the Treefrog ID of the descendant.
                    #lhalo_descendantID = np.where((treefrogID_to_lhaloID == treefrog_descendantID))[0] # This is the corresponding LHalo Number for the descendant.
                    lhalo_descendantID = treefrogID_to_lhaloID[treefrog_descendantID] # This is the corresponding LHalo Number for the descendant.
                    
                    if (isinstance(lhalo_descendantID, int) == False): # There can only be one descendant!
                        print("The Treefrog descendant ID is {0}.  This should correspond to an LHalo descendant ID of {1}".format(treefrog_descendantID, lhalo_descendantID))
                        print("The LHalo descendant ID should be a single integer.")
                        exit()
 
                    LHalos[lhalo_idx]['Descendant'] = lhalo_descendantID 
                    print("For LHalo Halo number {0} at snapshot {1}, the descendant is LHalo Halo Number {2} at snapshot {3}".format(lhalo_idx, snap_idx, LHalos[lhalo_idx]['Descendant'], descendant_snap))
                    print("This corresponds to Treefrog ID {0} at snapshot {1}, with a descendant ID {2} at snapshot {3}".format(f[snap_name]['ID'][treefrog_idx], snap_idx, treefrog_descendantID, descendant_snap)) 
                    exit() 
