from gym.envs.registration import register

try:
  register(
      id='corn_env-v0',
      entry_point='corn_env.envs:Learn_Corn',
  )
except:
  pass