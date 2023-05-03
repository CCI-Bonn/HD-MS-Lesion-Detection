"""                                                                                                            
:AUTHOR: Hagen Meredig                                                                                         
:ORGANIZATION: Department of Neuroradiology, Heidelberg Univeristy Hospital                                    
:CONTACT: Hagen.Meredig@med.uni-heidelberg.de                                                                  
:SINCE: August 18, 2021                            
"""

import os
import numpy as np


class AverageEnsemble(object):
    def __init__(self, out_dir: str = ""):
        self.out_dir = out_dir

    def __call__(self, epoch_data: dict, ident_key: any = None):

        accum = {}
        for data_part_key in epoch_data.keys():
            if data_part_key != ident_key:
                accum[data_part_key] = {}

        for data_part_key in epoch_data.keys():
            for entry_no in range(len(epoch_data[data_part_key])):
                if data_part_key != ident_key:
                    if ident_key == None:
                        ident = entry_no
                    else:
                        ident = epoch_data[ident_key][entry_no]
                    if not ident in accum[data_part_key].keys():
                        accum[data_part_key][ident] = [
                            np.array(epoch_data[data_part_key][entry_no]),
                            1,
                        ]
                    else:
                        accum[data_part_key][ident][0] += np.array(
                            epoch_data[data_part_key][entry_no]
                        )
                        accum[data_part_key][ident][1] += 1

        for data_part_key in epoch_data.keys():
            if data_part_key != ident_key:
                for ident in accum[data_part_key].keys():
                    accum[data_part_key][ident][0] /= accum[data_part_key][ident][1]

        ret = {}
        for data_part_key in accum.keys():
            ret[data_part_key] = []
            for ident in sorted(accum[data_part_key].keys()):
                ret[data_part_key].append(ret[data_part_key][ident][0])

        return ret

