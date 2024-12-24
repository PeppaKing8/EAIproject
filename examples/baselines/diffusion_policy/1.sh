python -m mani_skill.utils.download_demo "StackCube-v1"

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o rgbd \
  --save-traj --num-procs 10

for i in {0..9}; do
    # python train_rgbd.py --env-id StackCube-v1   --demo-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pos.cpu.h5   --control-mode "pd_ee_delta_pos" --sim-backend "gpu" --num-demos 100 --max_episode_steps 100   --total_iters 20000   --num-eval-envs 100 --seed $i --num_cams 2 --track
    python train_rgbd.py --env-id PickCube-v1   --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pos.cpu.h5   --control-mode "pd_ee_delta_pos" --sim-backend "gpu" --num-demos 100 --max_episode_steps 100   --total_iters 20000   --num-eval-envs 100 --seed $i --add_goal --track
done