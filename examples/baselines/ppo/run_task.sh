python ppo.py --env_id="HumanoidOpenDrawer-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=200_000_000 --eval_freq=10 --num-steps=500 \
  --control_mode="pd_joint_delta_pos" \
  --num_eval_steps=500 --checkpoint="/root/ManiSkill/examples/baselines/ppo/runs/HumanoidOpenDrawer-v1__ppo__1__1734790245/ckpt_1201.pt"