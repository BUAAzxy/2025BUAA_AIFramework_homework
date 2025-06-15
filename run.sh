export GLOG_v=0           
export FLAGS_call_stack_level=0   
python main_finetune.py \
  --data_path OCTID \
  --finetune RETFound_mae_meh \
  --epochs 50 --warmup_epochs 10 \
  --batch_size 16 --lr 5e-4 --layer_decay 0.65 \
  --task RETFound_mae_meh-OCTID --nb_classes 5

python main_finetune.py \
  --data_path Retina \
  --finetune RETFound_mae_meh \
  --epochs 50 --warmup_epochs 10 \
  --batch_size 16 --lr 5e-4 --layer_decay 0.65 \
  --task RETFound_mae_meh-Retina --nb_classes 4

python main_finetune.py \
  --data_path APTOS2019 \
  --finetune RETFound_mae_meh \
  --epochs 50 --warmup_epochs 10 \
  --batch_size 16 --lr 5e-4 --layer_decay 0.65 \
  --task RETFound_mae_meh-APTOS2019 --nb_classes 5

python main_finetune.py \
  --data_path JSIEC \
  --finetune RETFound_mae_meh \
  --epochs 50 --warmup_epochs 10 \
  --batch_size 16 --lr 5e-4 --layer_decay 0.65 \
  --task RETFound_mae_meh-JSIEC --nb_classes 39