from train_setup import *


replay_buffer = np.load('../../random_data_100K.npy', allow_pickle=True)

episode_xs = []
episode_ys = []
for i in range(len(replay_buffer)):
  print('traj', i)
  episode_x = [x[1] for x in replay_buffer[i]]
  episode_y = [x[2] for x in replay_buffer[i]]
  if len(episode_x)>10 and episode_x[0] is not None:
    episode_xs.append(episode_x)
    episode_ys.append(episode_y) 
print('Load %d episodes!'%len(episode_xs))

def data_generator():
  while True:
    idx = np.random.randint(len(episode_xs))
    x = episode_xs[idx]
    y = episode_ys[idx]
    MAX_CONTINUOUS_PLAY = len(x)
    first_second_pairs = []
    current_first = 0
    while True:
      distance = random.randint(1, MAX_ACTION_DISTANCE)
      second = current_first + distance
      if second >= MAX_CONTINUOUS_PLAY:
        break
      first_second_pairs.append((current_first, second))
      current_first = second + 1
    random.shuffle(first_second_pairs)
    x_result = []
    y_result = []
    for first, second in first_second_pairs:
      future_x = x[second]
      current_x = x[first]
      previous_x = current_x
      if first > 0:
        previous_x = x[first - 1]
      current_y = y[first]
      x_result.append(np.concatenate((previous_x, current_x, future_x), axis=2))
      y_result.append(current_y)
      if len(x_result) == BATCH_SIZE:
        yield (np.array(x_result),
               keras.utils.to_categorical(np.array(y_result),
                                          num_classes=ACTION_CLASSES))
        x_result = []
        y_result = []

if __name__ == '__main__':
  EXPERIMENT_OUTPUT_FOLDER = 'beenchwood1M_R'
  logs_path, current_model_path = setup_training_paths(EXPERIMENT_OUTPUT_FOLDER)
  model = ACTION_NETWORK(((1 + ACTION_STATE_ENCODING_FRAMES) * NET_CHANNELS, NET_HEIGHT, NET_WIDTH), ACTION_CLASSES)
  adam = keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
  callbacks_list = [keras.callbacks.TensorBoard(log_dir=logs_path, write_graph=False),
                    keras.callbacks.ModelCheckpoint(current_model_path,
                                                    period=MODEL_CHECKPOINT_PERIOD)]
  model.fit_generator(data_generator(),
                      steps_per_epoch=DUMP_AFTER_BATCHES,
                      epochs=ACTION_MAX_EPOCHS,
                      callbacks=callbacks_list)
