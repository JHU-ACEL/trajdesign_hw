import numpy as np
import matplotlib.pyplot as plt

def plot_line_segments(segments, **kwargs):
  plt.plot([x for tup in [(p1[0], p2[0], None) for (p1, p2) in segments] for x in tup],
           [y for tup in [(p1[1], p2[1], None) for (p1, p2) in segments] for y in tup], **kwargs)

def plot_boxes(boxes, **kwargs):
  for obs in boxes:
    rect = plt.Rectangle((obs[0], obs[2]), \
                              obs[1]-obs[0], obs[3]-obs[2], \
                             fc='red', ec='blue')
    plt.gca().add_patch(rect)

def ccw(A, B, C):
  return np.cross(B - A, C - A) > 0

def line_line_intersection(l1, l2):
  A, B = np.array(l1)
  C, D = np.array(l2)
  return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def obs_intersect(obs_1, obs_2):
  intersect = True
  if obs_1[1] < obs_2[0] or \
     obs_2[1] < obs_1[0] or \
     obs_1[3] < obs_2[2] or \
     obs_2[3] < obs_1[2]:
    intersect = False
  return intersect

def is_free_state(x0, obstacles):
  """ Check if x0 is free state given list of obstacles"""
  if any([x0[0] >= obstacle[0] and x0[0] <= obstacle[1] and \
           x0[1] >= obstacle[2] and x0[1] <= obstacle[3] \
           for obstacle in obstacles]):
    return False
  return True

def find_obs(x0, n_obs, posmin, posmax, \
             border_size, box_buffer, min_box_size, max_box_size, \
             max_iter=100, ignore_intersection=True):
  """ Given state x0, place obstacles between x0 and posmax"""
  obs = []
  itr = 0
  while len(obs) < n_obs and itr < max_iter:
    xmin = (posmax[0] - border_size - max_box_size)*np.random.rand() + border_size
    xmin = np.max([xmin, x0[0]])
    xmax = xmin + min_box_size  + (max_box_size - min_box_size)*np.random.rand()
    ymin = (posmax[1] - border_size - max_box_size)*np.random.rand() + border_size
    ymin = np.max([ymin, x0[1]])
    ymax = ymin + min_box_size  + (max_box_size - min_box_size)*np.random.rand()
    obstacle = np.array([xmin - box_buffer, xmax + box_buffer, \
                    ymin - box_buffer, ymax + box_buffer])

    intersecting = False
    for obs_2 in obs:
      intersecting = obs_intersect(obstacle, obs_2)
      if intersecting:
        break
    if ignore_intersection:
      intersecting = False

    if is_free_state(x0, [obstacle]) and not intersecting:
      obs.append(obstacle)
    itr += 1

  if len(obs) is not n_obs:
    return []
  return obs

def findIC(obstacles, posmin, posmax, velmin, velmax, max_iter=1000):
  """ Given list of obstacles, find IC that is collision free"""
  IC_found = False
  itr = 0
  while not IC_found and itr < max_iter:
    r0 = posmin + (posmax-posmin)*np.random.rand(2)
    if not any([r0[0] >= obstacle[0] and r0[0] <= obstacle[1] and \
         r0[1] >= obstacle[2] and r0[1] <= obstacle[3] \
         for obstacle in obstacles]):
      IC_found = True
  if not IC_found:
    return np.array([])
  x0 = np.hstack((r0, velmin + (velmax-velmin)*np.random.rand(2)))
  return x0

def random_obs(n_obs, posmin, posmax, border_size:float=0.05, box_buffer:float=0.025, \
              min_box_size:float=0.25, max_box_size:float=0.75, max_iter:int=100):
  """ Generate random list of obstacles in workspace """
  obstacles = []
  itr = 0
  while itr < max_iter and len(obstacles) is not n_obs:
    # Valid range for rectangle placement
    x_range = posmax[0] - posmin[0] - border_size - max_box_size
    y_range = posmax[1] - posmin[1] - border_size - max_box_size

    # Generate random position within valid range
    xmin = posmin[0] + border_size + x_range * np.random.rand()
    ymin = posmin[1] + border_size + y_range * np.random.rand()

    # Generate random size
    box_size_x = min_box_size + (max_box_size - min_box_size) * np.random.rand()
    box_size_y = min_box_size + (max_box_size - min_box_size) * np.random.rand()

    xmax = xmin + box_size_x
    ymax = ymin + box_size_y

    obstacle = np.array([xmin - box_buffer, xmax + box_buffer, \
                        ymin - box_buffer, ymax + box_buffer])

    intersecting = False
    for obs_2 in obstacles:
      intersecting = obs_intersect(obstacle, obs_2)
      if intersecting:
        break
    if not intersecting:
      obstacles.append(obstacle)
    itr += 1

  if len(obstacles) is not n_obs:
    obstacles = []
  return obstacles
