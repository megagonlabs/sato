# feature importance
python feature_importance.py  --model_type=single --model_path=sherlock_None.pt
python feature_importance.py  --model_type=single --model_path=all_None.pt --topic=num-directstr_thr-0_tn-400

python feature_importance.py  --model_type=CRF --model_path=CRF_pre.pt 
python feature_importance.py  --model_type=CRF --model_path=CRF+LDA_pre.pt --topic=num-directstr_thr-0_tn-400


cd $BASEPATH/model
python train_CRF_LC.py -c params/crf_configs/CRF.txt --multi_col_only=true --comment=path 
python train_CRF_LC.py -c params/crf_configs/CRF+LDA.txt --multi_col_only=true --comment=pathL 

python train_CRF_LC.py -c params/crf_configs/CRF.txt  --comment=path 
python train_CRF_LC.py -c params/crf_configs/CRF+LDA.txt  --comment=pathL 

cd $BASEPATH/scripts
python per_type.py