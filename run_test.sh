!/bin/bash

#----------------------------------------------------------------
#----------------------------------------------------------------
# --- (mpe-pp) ---

# 112358, 42, 999, 123, 2024

# poam
# python src/main.py --config=open/open_train_pp \
# --alg-config=mpe/poam \
# --env-config=mpe \
# with t_max=3000000 \
# env_args.key="mpe:PredatorPrey-v0" \
# env_args.time_limit=100 \
# env_args.pretrained_wrapper="PretrainedTag" \
# ed_pretrain_ts=500000 \
# --seed=123

# IFS
# python src/main.py --config=open/ifs/open_train_pp_ifs \
# --alg-config=mpe/ifs \
# --env-config=mpe \
# with t_max=3000000 \
# env_args.key="mpe:PredatorPrey-v0" \
# env_args.time_limit=100 \
# env_args.pretrained_wrapper="PretrainedTag" \
# ed_pretrain_ts=500000 \
# --seed=112358


#-----------------------------------------------
# --- (SMAC 5v6) ---

# poam
# python src/main.py --config=open/open_train_5v6 \
# --alg-config=sc2/poam \
# --env-config=sc2 \
# with t_max=5000000 \
# env_args.map_name=5m_vs_6m \
# ed_pretrain_ts=1000000 \
# --seed=112358

#-----------------------------------------------
# --- (SMAC 8v9) ---

# poam
# python src/main.py --config=open/open_train_8v9 \
# --alg-config=sc2/poam \
# --env-config=sc2 \
# with t_max=5000000 \
# env_args.map_name=8m_vs_9m \
# ed_pretrain_ts=1000000 \
# --seed=112358


#-----------------------------------------------
# --- (SMAC 10v11) ---

# poam
# python src/main.py --config=open/open_train_10v11 \
# --alg-config=sc2/poam \
# --env-config=sc2 \
# with t_max=5000000 \
# env_args.map_name=10m_vs_11m \
# ed_pretrain_ts=1000000 \
# --seed=112358


#-----------------------------------------------
# --- (SMAC 3sv5z) ---

# poam
# python src/main.py --config=open/open_train_3sv5z \
# --alg-config=sc2/poam \
# --env-config=sc2 \
# with t_max=5000000 \
# env_args.map_name=3s_vs_5z \
# ed_pretrain_ts=1000000 \
# --seed=112358


#----------------------------------------------------------------
#----------------------------------------------------------------
