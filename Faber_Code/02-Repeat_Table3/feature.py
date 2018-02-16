#---- file feature.py
#       All the functions for feature vector exportation.
#----

#---- function parse_feature_CM()
#   - Input:
#       CM_Path: string
#         CM feature vector file provided by Faber et al. (2017)
#   - Output:
#       CM_vec: array(133885, 900)
#         CM feature vector, not normalized yet
#         length of CM_vec is pre-defined
#   - Variables:
#       CM_file: changeable
#         temporary stroage of CM file
#   ! Note:
#     !  1. Data is stored in 133885 length, which is larger than actully tested
#           molecule number 130829. so returned CM_vec has some rows completely
#           filled with zeros.
#----
def parse_feature_CM(CM_Path):
    import numpy as np
    CM_vec = np.zeros([133885, 900])
    CM_file = open(CM_Path).read().split('\n')
    for i in CM_file[:-1]:
        qm9_entry = i.split()
        qm9_ind = int(qm9_entry[0].replace("qm9:",""))
        for Indj, j in enumerate(qm9_entry[1:]):
            CM_vec[qm9_ind-1][Indj] = float(j)
    return CM_vec

