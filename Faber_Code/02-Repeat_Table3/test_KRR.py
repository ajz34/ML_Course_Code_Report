#! /share/home/zyzhu/anaconda3/bin/python

#---- file import
import util
import feature
import fold
import prop
import fit
#---- 

#---- main program
#       The main program should be able to predict all properties of KRR.
#----
def mainprog():

#-- Test functions

#-- Test of parse_feature_CM
#   import numpy as np
#   import time
#   CM_Path = "/share/home/zyzhu/Documents-Graduate/Faber-Lilienfeld.JCTC.2017.ASAP/SI_Source/Feature_Files/CM"
#   time0 = time.time()
#   CM_vec = feature.parse_feature_CM(CM_Path)
#   print(CM_vec[1])
#   print("Execuation time: %s" % (time.time() - time0))

#-- Test of parse_prop
#   import numpy as np
#   QM9_Prop, QM9_Index = parse_prop("/share/home/zyzhu/Documents-Graduate/Faber-Lilienfeld.JCTC.2017.ASAP/SI_Source/Prop_Files/qm9-mol-info-standardized-v1")
#   print(QM9_Index[:100])
#   print(QM9_Prop[:5])

#-- Test of predefined_fold
#   Predefined_Split, Train_Index, Test_Index = fold.predefined_fold()

    import numpy as np
    import time
    #  0. output file
    FOut = open("CM-KRR.txt", "w")
    time0 = time.time()
    #  1. CM vector parsing
    CM_Path = "/share/home/zyzhu/Documents-Graduate/Faber-Lilienfeld.JCTC.2017.ASAP/SI_Source/Feature_Files/CM"
    CM_vec = feature.parse_feature_CM(CM_Path)
    # print(CM_vec[1])
    time1 = time.time()
    FOut.write("Execuation time 01 - CM vector parsing: %s \n" % (time1 - time0))
    #  2. Property parsing
    QM9_Prop, QM9_Index = prop.parse_prop("/share/home/zyzhu/Documents-Graduate/Faber-Lilienfeld.JCTC.2017.ASAP/SI_Source/Prop_Files/qm9-mol-info-standardized-v1")
    time2 = time.time()
    FOut.write("Execuation time 02 - Property parsing: %s \n" % (time2 - time1))
    #  3. Fold parsing
    Predefined_Split, Train_Index, Test_Index = fold.predefined_fold()
    time3 = time.time()
    FOut.write("Execuation time 03 - Fold parsing: %s \n" % (time3 - time2))
    #  4. Reindex to training and testing set
    # Only test 10% testing set. The result should be able to be compared to Table 3 in
    # Faber et al. (JCTC2017)
    # For this file, all the properties are tested.
    # ! Note that CM_vec is 133885 length, so no reindex needed to this vector.
    #   However, some rows in CM_vec is empty.
    #   QM9_Prop as well.
    # ! The 10% training and testing set ([0]) should be created with reindex process.
    # ! *_Y as property arrays are indexed as [Prop, reindexed-Id].
    # - Train_X
    # - Train_Y
    # - Test_X
    # - Test_Y
    # - Temp_prop_U: Temporary extraction of the property U0
    Train_X = np.zeros([len(Train_Index[0]), CM_vec.shape[1]])
    Train_Y = np.zeros([13, len(Train_Index[0])])
    Test_X = np.zeros([len(Test_Index[0]), CM_vec.shape[1]])
    Test_Y = np.zeros([13, len(Test_Index[0])])
    for Indi, i in enumerate(Train_Index[0]):
        Train_X[Indi] = CM_vec[i-1]
        Train_Y[:,Indi] = QM9_Prop[i-1,:]
    for Indi, i in enumerate(Test_Index[0]):
        Test_X[Indi] = CM_vec[i-1]
        Test_Y[:,Indi] = QM9_Prop[i-1,:]
    time4 = time.time()
    FOut.write("Execuation time 04 - Reindex: %s \n" % (time4 - time3))
    #  5. Training and Testing - KRR
    FOut.write("\n")
    FOut.write("--- Training and Testing BR ---\n")
    FOut.write("\n")
    FOut.write(" Prop_Id" + "       Err_MAD" + "      Err_RMSD" + " Time_Train" + "  Time_Test" + "\n" )
    for Indi in range(13):
        (Err_MAD, Err_RMSD, Time_Train, Time_Test) = \
            fit.fit_KRR(Train_X, Train_Y[Indi], Test_X, Test_Y[Indi], Predefined_Split[0])
        FOut.write("{:8}{:14.8f}{:14.8f}{:11.2f}{:11.2f}\n".format(Indi, Err_MAD, Err_RMSD, Time_Train, Time_Test))

#---- dummy main

#---- dummy main
#----
mainprog()
