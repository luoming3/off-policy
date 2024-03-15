#!/bin/sh
env="custom"
scenario="shared_7a"
num_landmarks=3
num_agents=4
algo="qmix"
exp="el1024_bas32"
seed_max=1

current_dir=$(cd $(dirname $0); pwd)

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=7 python -u ${current_dir}/train/train_custom.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --episode_length 1024 \
    --batch_size 32 --lr 5e-4 --num_env_steps 300000000 \
    --n_rollout_threads 1 --buffer_size 5000 --save_interval 50000 --epsilon_anneal_time 500000 \
    --train_interval_episode 1 --num_eval_episodes 1 --use_wandb
    echo "training is done!"
done
