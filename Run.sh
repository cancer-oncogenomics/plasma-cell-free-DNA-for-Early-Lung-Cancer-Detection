# Layer 1: Model Training
./main.py LungModel         --feature  tests/feature/MS.csv         --train_info  tests/Train_info.list         --pred_info  tests/Valid_info.list         --d_output  layer1/MS/         --prefix  MS         --nthreads   10
./main.py LungModel         --feature  tests/feature/CNV.csv         --train_info  tests/Train_info.list         --pred_info  tests/Valid_info.list         --d_output  layer1/CNV/         --prefix  CNV         --nthreads   10
./main.py LungModel        --feature  tests/feature/FSD.csv         --train_info  tests/Train_info.list         --pred_info  tests/Valid_info.list         --d_output  layer1/FSD/         --prefix  FSD         --nthreads   10

# Base Model ROC Calculation
Rscript ./basemodel.roc.cal.R --info tests/ --automl_dir layer1/ --fout basemodel.auc.summary.txt

# Layer 2
Rscript ./featureCombn.R --top_n 5 --base_auc_dir layer1/
