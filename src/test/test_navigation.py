import numpy as np
from sptm import *

action_model = load_keras_model(3, ACTION_CLASSES, '../../experiments/demo_L/models/model.019000.h5')
memory = SPTM()
memory.set_shortcuts_cache_file('beenchwood')

replay_buffer = np.load('../../random_data_100K.npy', allow_pickle=True)

keyframes = []
keyframe_coordinates = []
keyframe_actions = []
for i in range(len(replay_buffer)):
  print('traj', i)
  episode = replay_buffer[i]
  if len(episode)>10 and episode[0] is not None:
    for j in range(len(episode)):
      keyframes.append(episode[j][1])
      keyframe_coordinates.append(episode[j][0])
      keyframe_actions.append(episode[j][2])

keyframes = keyframes[:3000]
keyframe_coordinates = keyframe_coordinates[:3000]
memory.build_graph(keyframes, keyframe_coordinates)

from gibson.envs.husky_env import HuskyNavigateEnv
env = HuskyNavigateEnv(config='/home/guoyijie/imitation-navigation/gibson_configs/beechwood_c0_rgb_skip50_random_separate.yaml')
import sys
sys.path.append('../../')
from wrappers import WrapState
env = WrapState(env, state_mode='xyz+rpy', ob_mode='rgb_filled', imsize=128)

TEST_REPEAT=1
reward_list = []
for trial_idx in range(100):
  ob = env.reset()
  state = env.get_state()   
  goal_location, goal_frame = env.get_target_image() 
  best_index, best_probability = memory.set_goal(goal_frame, goal_location, keyframe_coordinates)
  keyframes.append(goal_frame)
  keyframe_coordinates.append(goal_location) 
  memory.compute_shortest_paths(len(keyframes) - 1)  
  goal_localization_keyframe_index = best_index
  done = False
  reward = 0
  just_started = True
  screens = []
  coordinates = []
  while not done:
    target_index, nn = memory.find_intermediate_reachable_goal(ob, state, keyframe_coordinates)
    if target_index is None:
      target_index = len(keyframes) - 1
      not_localized_count = 1
    else:
      not_localized_count = 0
    screens.append(ob)
    coordinates.append(state)
    if not_localized_count == 0:
      #align step
      if just_started:
        for _ in range(TEST_REPEAT):
          screens.append(ob)
          coordinates.append(state)
        just_started = False
      first_arg = screens[-1 - TEST_REPEAT]
      second_arg = screens[-1]
      third_arg = keyframes[target_index]
      x = np.expand_dims(np.concatenate((first_arg,
                                         second_arg,
                                         third_arg), axis=2), axis=0)
      action_probabilities = np.squeeze(action_model.predict(x, batch_size=1))
      action_index = np.random.choice(4, p=action_probabilities)
      for repeat_index in range(TEST_REPEAT):
        ob, r, done, info = env.step(action_index)
        state = info['state']
        screens.append(ob)
        coordinates.append(state)
        reward += r
        print('reward', reward)
    else:
      # random step
      for counter in range(1):
        action_index = random.randint(0, 4)
        for repeat_index in range(TEST_REPEAT):
          ob, r, done, info = env.step(action_index)
          state = info['state']
          screens.append(ob)
          coordinates.append(state)
          reward += r
          print('reward', reward)
  reward_list.append(reward)

print(reward_list)
print('average reward', np.mean(reward_list))
