import pandas as pd

KB8Y = pd.read_csv("/mnt/binf/stephanie/Mercury/KB8Y-github/github/tests/KB8Y_usedinfo.csv")
train = pd.read_table("/mnt/binf/stephanie/Mercury/KB8Y-github/github/tests/Train.info.list")
valid = pd.read_table("/mnt/binf/stephanie/Mercury/KB8Y-github/github/tests/Valid.info.list")

cnv = pd.read_csv("~/Mercury/KB8Y-github/github/tests/Feature/cnv.csv")
frag_arm = pd.read_csv("~/Mercury/KB8Y-github/github/tests/Feature/frag.arm2023.csv")
MCMS = pd.read_csv("~/Mercury/KB8Y-github/github/tests/Feature/MCMS.csv")


# train subset
# subset random 50 samples
train_subset = train.sample(50)
train_subset['ID'] = ['P' + str(i) for i in range(1, 51)]

cnv_train_subset = pd.merge(train_subset[['ID','SampleID']], cnv, on=["SampleID"], how="left")
cnv_train_subset.drop(columns=["SampleID"], inplace=True)
cnv_train_subset.to_csv("/mnt/binf/stephanie/Mercury/KB8Y-github/github/tests/feature/CNV_train.csv", index=False)

frag_arm_train_subset = pd.merge(train_subset[['ID','SampleID']], frag_arm, on=["SampleID"], how="left")
frag_arm_train_subset.drop(columns=["SampleID"], inplace=True)
frag_arm_train_subset.to_csv("/mnt/binf/stephanie/Mercury/KB8Y-github/github/tests/feature/FSD_train.csv", index=False)


MCMS_train_subset = pd.merge(train_subset[['ID','SampleID']], MCMS, on=["SampleID"], how="left")
MCMS_train_subset.drop(columns=["SampleID"], inplace=True)
SBS_col = [col for col in MCMS_train_subset.columns if "SBS" in col]
MCMS_train_subset = MCMS_train_subset[['ID'] + SBS_col]
MCMS_train_subset.to_csv("/mnt/binf/stephanie/Mercury/KB8Y-github/github/tests/feature/MS_train.csv", index=False)

train_subset = pd.merge(train_subset, KB8Y, on=["SampleID"], how="left")
# replace the missing values by "-"
train_subset.fillna("-", inplace=True)
train_subset.rename(columns={"Age_x": "Age",
                             'Sex_x':"Sex",}, inplace=True)
train_subset.drop(columns=["SampleID"], inplace=True)
train_subset.rename(columns={"ID": "SampleID"}, inplace=True)
selected_columns = ['SampleID', 'Response', "Age","Sex","Response","Train_Group","TumorSize","NoduleNumber","Histology","ADCsubtype"]
train_subset = train_subset[selected_columns]
train_subset.to_csv("/mnt/binf/stephanie/Mercury/KB8Y-github/github/tests/Train_info.list", sep="\t", index=False)

# valid subset 
valid_subset = valid.sample(50)
valid_subset['ID'] = ['V' + str(i) for i in range(1, 51)]

cnv_valid_subset = pd.merge(valid_subset[['ID','SampleID']], cnv, on=["SampleID"], how="left")
cnv_valid_subset.drop(columns=["SampleID"], inplace=True)
cnv_valid_subset.to_csv("/mnt/binf/stephanie/Mercury/KB8Y-github/github/tests/feature/CNV_valid.csv", index=False)

frag_arm_valid_subset = pd.merge(valid_subset[['ID','SampleID']], frag_arm, on=["SampleID"], how="left")
frag_arm_valid_subset.drop(columns=["SampleID"], inplace=True)
frag_arm_valid_subset.to_csv("/mnt/binf/stephanie/Mercury/KB8Y-github/github/tests/feature/FSD_valid.csv", index=False)


MCMS_valid_subset = pd.merge(valid_subset[['ID','SampleID']], MCMS, on=["SampleID"], how="left")
MCMS_valid_subset.drop(columns=["SampleID"], inplace=True)
SBS_col = [col for col in MCMS_valid_subset.columns if "SBS" in col]
MCMS_valid_subset = MCMS_valid_subset[['ID'] + SBS_col]
MCMS_valid_subset.to_csv("/mnt/binf/stephanie/Mercury/KB8Y-github/github/tests/feature/MS_valid.csv", index=False)

valid_subset = pd.merge(valid_subset, KB8Y, on=["SampleID"], how="left")
valid_subset.fillna("-", inplace=True)
valid_subset.rename(columns={"Age_x": "Age",
                             'Sex_x':"Sex",}, inplace=True)
valid_subset.drop(columns= ['SampleID'], inplace=True)
valid_subset.rename(columns={"ID": "SampleID"}, inplace=True)
selected_columns = ['SampleID', 'Response', "Age","Sex","Response","Train_Group","TumorSize","NoduleNumber","Histology","ADCsubtype"]
valid_subset = valid_subset[selected_columns]
valid_subset.to_csv("/mnt/binf/stephanie/Mercury/KB8Y-github/github/tests/Valid_info.list", sep="\t", index=False)

# train and valid feature combine as one
CNV_subset = pd.concat([cnv_train_subset, cnv_valid_subset], ignore_index=True)
CNV_subset.rename(columns={"ID": "SampleID"}, inplace=True)
CNV_subset.to_csv("/mnt/binf/stephanie/Mercury/KB8Y-github/github/tests/feature/CNV.csv", index=False)

FSD_subset = pd.concat([frag_arm_train_subset, frag_arm_valid_subset], ignore_index=True)
FSD_subset.rename(columns={"ID": "SampleID"}, inplace=True)
FSD_subset.to_csv("/mnt/binf/stephanie/Mercury/KB8Y-github/github/tests/feature/FSD.csv", index=False)

MS_subset = pd.concat([MCMS_train_subset, MCMS_valid_subset], ignore_index=True)
MS_subset.rename(columns={"ID": "SampleID"}, inplace=True)
MS_subset.to_csv("/mnt/binf/stephanie/Mercury/KB8Y-github/github/tests/feature/MS.csv", index=False)
