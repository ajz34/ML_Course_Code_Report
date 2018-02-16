#---- prop.py
#       Parse properties, or unit conversion utilities specified to QM9 dataset.
#----

#---- function parse_prop
#       Parse standardized QM9 database molecule properties.
#       The file with the input path should be provided by Faber et al. (2017)
#   - Input:
#       Prop_Path: string
#         Standardized QM9 database molecule property file provided by Faber et al. (2017)
#   - Output: (tuple)
#       QM9_Prop: array(133885, 13)
#         Standardized QM9 database molecule properties.
#       QM9_Index: array(133885)
#         Index of the actually tested molecules.
#   - Variables:
#       QM9_Prop_File: changeable
#         temporary storage of QM9 property file.
#----
def parse_prop(Prop_Path):
    import numpy as np
    QM9_Prop_File = open(Prop_Path).read().split('\n')[1:-1]
    QM9_Prop = np.zeros([133885, 13])
    QM9_Index = np.zeros(133885)
    for Indi, i in enumerate(QM9_Prop_File):
        temp_strlst = i.split()
        QM9_Index[Indi] = int(temp_strlst[0].replace("qm9:",""))
        for Indj in range(13):
            QM9_Prop[int(temp_strlst[0][4:])-1][Indj] = float(temp_strlst[Indj+2])
    return (QM9_Prop, QM9_Index)

