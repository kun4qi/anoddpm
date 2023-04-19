#!/bin/bash

#$-l rt_G.large=1
#$ -l h_rt=72:00:00
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
module load gcc/12.2.0 python/3.10/3.10.10 cuda/11.6 cudnn/8.4

~/brats/bin/python3 train_anoddpm_lightning.py -c configs/config_train_anoddpm.json -s
