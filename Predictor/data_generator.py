import random
from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd
import copy
import os
import re
import configparser

def new_calculation_flight(T, start_t, delta_t, start_boost, start_speed, start_X, start_Y, start_Z, start_angle, max_angle_change, max_angle, max_speed, max_pos_boost, max_neg_boost, min_speed, prob_boost_change, prob_angle_change, max_height_speed, mid_height):
  begining_time = start_t
  time_getting_height_and_downing = (mid_height/max_height_speed)
  old_Z = start_Z
  new_Z = 0
  N = int(T/delta_t)
  N_downing = (time_getting_height_and_downing/delta_t)
  old_X=start_X
  new_X = 0
  old_Y=start_Y
  new_Y = 0
  old_angle = start_angle
  new_angle = 0
  old_speed = start_speed
  new_speed = 0
  old_boost = start_boost
  new_boost = 0

  trajectory_vect = []

  upping = True
  downing = False

  for j in range(N-1):
    weight_angle = 1/(abs(old_angle+1)/(max_angle))
    prob_boost = [0+prob_boost_change/2, 1-prob_boost_change/2]
    prob_angle = [0+(prob_angle_change/2)/weight_angle, 1-(prob_angle_change/2)*weight_angle]

    trajectory_vect.append( (old_X, old_Y, old_Z, start_t) )
    start_t+=delta_t

    rr = random.random()
    if(rr<prob_angle[0]):
      new_angle = old_angle - (max_angle_change * (rr/prob_angle[0]))
    elif(rr>prob_angle[1]):
      new_angle = old_angle + (max_angle_change * ((rr-prob_angle[1])/(1-prob_angle[1])))

    rr = random.random()
    if(rr<prob_boost[0]):
      new_boost = old_boost - max_neg_boost*(rr/prob_boost[0])
      if(new_boost<max_neg_boost):
        new_boost = max_neg_boost
    elif(rr>prob_boost[1]):
        new_boost = old_boost + max_pos_boost*(rr/prob_boost[1])
        if(new_boost>max_pos_boost):
            new_boost = max_pos_boost

    new_speed = old_speed + delta_t * old_boost
    if(new_speed<min_speed):
        new_speed=min_speed
    if(new_speed>max_speed):
        new_speed=max_speed

    new_X = old_X + (old_speed*math.cos( math.radians(new_angle) ))*delta_t + (old_boost*math.cos( math.radians(new_angle) ))*delta_t*delta_t/2
    new_Y = old_Y + (old_speed*math.sin( math.radians(new_angle) ))*delta_t + (old_boost*math.sin( math.radians(new_angle) ))*delta_t*delta_t/2
    
    if((new_Z<mid_height) and upping):
        new_Z = old_Z + max_height_speed*delta_t
    
    if(new_Z>mid_height):
        upping=False
        
    if(N-j-1 <= N_downing):
        downing=True
        
    if(downing):
        new_Z = old_Z - max_height_speed*delta_t
    
    old_X = new_X
    old_Y = new_Y
    old_Z = new_Z
    old_speed = new_speed
    old_angle = new_angle
    old_boost = new_boost

  return trajectory_vect

def inaccuracy_maker(np_trajectory, prob_inaccuracy, inaccuracy_wide_x, inaccuracy_wide_y, is_it_pecents=False, percent = 0):
  prob = [0+prob_inaccuracy/2, 1-prob_inaccuracy/2]

  for i in range(np_trajectory.shape[0]):
    if(is_it_pecents):
      rr = random.random()
      if(rr<prob[0]):
        np_trajectory[i][0] -= percent * np_trajectory[i][0] * (rr/prob[0])
      elif(rr>prob[1]):
        np_trajectory[i][0] += percent * np_trajectory[i][0] * ((rr-prob[1])/(1-prob[1]))

      rr = random.random()
      if(rr<prob[0]):
        np_trajectory[i][1] -= percent * np_trajectory[i][1] * (rr/prob[0])
      elif(rr>prob[1]):
        np_trajectory[i][1] += percent * np_trajectory[i][1] * ((rr-prob[1])/(1-prob[1]))
    else:
      rr = random.random()
      if(rr<prob[0]):
        np_trajectory[i][0] -= inaccuracy_wide_x* (rr/prob[0])
      elif(rr>prob[1]):
        np_trajectory[i][0] += inaccuracy_wide_x* ((rr-prob[1])/(1-prob[1]))

      rr = random.random()
      if(rr<prob[0]):
        np_trajectory[i][1] -= inaccuracy_wide_y* (rr/prob[0])
      elif(rr>prob[1]):
        np_trajectory[i][1] += inaccuracy_wide_y* ((rr-prob[1])/(1-prob[1]))

  return np.array(np_trajectory)


def dubbling_maker(np_trajectory, prob_dubbling):
  prob = [0+prob_dubbling/2, 1-prob_dubbling/2]
  new_np_trajectory=[]
  for i in range(np_trajectory.shape[0]):
    rr = random.random()
    if(prob[0]>rr or prob[1]<rr):
      new_np_trajectory.append(np_trajectory[i])
    new_np_trajectory.append(np_trajectory[i])
  return np.array(new_np_trajectory)

def timeshift_maker(np_trajectory, prob_timeshift, timeshift_wide):
  prob = [0+prob_timeshift/2, 1-prob_timeshift/2]

  for i in range(np_trajectory.shape[0]):
    rr = random.random()
    if(rr<prob[0]):
      np_trajectory[i][3] -= timeshift_wide
    elif(rr>prob[1]):
      np_trajectory[i][3] += timeshift_wide

  return np.array(np_trajectory)

def missing_maker(np_trajectory, prob_missing):
  prob = [0+prob_missing/2, 1-prob_missing/2]
  new_np_trajectory=[]
  for i in range(np_trajectory.shape[0]):
    rr = random.random()
    if(prob[0]<rr and prob[1]>rr):
      new_np_trajectory.append(np_trajectory[i])

  return np.array(new_np_trajectory)

def slope_zero(new_np_trajectory):
  start_point_ray_y = new_np_trajectory[0][1]
  start_point_ray_x = new_np_trajectory[0][0]
  end_point_ray_y = new_np_trajectory[-1][1]
  end_point_ray_x = new_np_trajectory[-1][0]
  vector_ray = [end_point_ray_x - start_point_ray_x,  end_point_ray_y - start_point_ray_y]
  length_ray = math.sqrt(math.pow(vector_ray[0],2) + math.pow(vector_ray[1],2))
  tang_ray = vector_ray[1] / vector_ray[0]
  angle_rad_ray = math.atan(tang_ray)

  for i in range(new_np_trajectory.shape[0]):
    start_point_y = new_np_trajectory[0][1]
    start_point_x = new_np_trajectory[0][0]
    end_point_y = new_np_trajectory[i][1]
    end_point_x = new_np_trajectory[i][0]
    vector_point = [end_point_x-start_point_x, end_point_y - start_point_y]
    length_point = math.sqrt(math.pow(vector_point[0], 2) + math.pow(vector_point[1], 2))
    tang_point = new_np_trajectory[i][1] / new_np_trajectory[i][0]
    angle_rad_point = math.atan(tang_point)
    angle_rad_diff = angle_rad_point - angle_rad_ray
    new_np_trajectory[i][1] = length_point*math.sin(angle_rad_diff)
    new_np_trajectory[i][0] = length_point*math.cos(angle_rad_diff)

  return new_np_trajectory

def save_generated_data(result_df):
  path = "./saved_data"
  files = os.listdir (path)
  max_num_file = 0
  for tmp in files:
    if (re.match("output_\d+.xlsx",tmp)):
      m = re.search(r'_(\w+).', tmp)
      current_num = int(m.group(1))
      if (max_num_file < current_num):
        max_num_file = current_num
  print(files)
  result_succ = result_df.to_excel(path + "/"+f"output_{max_num_file+1}.xlsx")
  if(result_succ):
    return True
  else:
    return False


def main():

    config = configparser.ConfigParser()
    config.read("settings.ini")

    starting_val = float(config["DataGenerator"]["starting_val"])
    T = float(config["DataGenerator"]["T"])
    delta_t = float(config["DataGenerator"]["delta_t"])

    start_hour = float(config["DataGenerator"]["start_hour"])
    start_minutes = float(config["DataGenerator"]["start_minutes"])
    start_second = float(config["DataGenerator"]["start_second"])
    start_time = start_hour*3600 + start_minutes*60 + start_second
    X_start = float(config["DataGenerator"]["X_start"])
    Y_start = float(config["DataGenerator"]["Y_start"])
    Z_start = float(config["DataGenerator"]["Z_start"])
    max_pos_boost = float(config["DataGenerator"]["max_pos_boost"])
    max_neg_boost = float(config["DataGenerator"]["max_neg_boost"])
    initial_boost = float(config["DataGenerator"]["initial_boost"])
    min_speed = float(config["DataGenerator"]["min_speed"])
    max_speed = float(config["DataGenerator"]["max_speed"])
    initial_speed = float(config["DataGenerator"]["initial_speed"])
    initial_angle = float(config["DataGenerator"]["initial_angle"])
    max_angle_change = float(config["DataGenerator"]["max_angle_change"])
    max_angle = float(config["DataGenerator"]["max_angle"])
    prob_boost_change = float(config["DataGenerator"]["prob_boost_change"])
    prob_angle_change = float(config["DataGenerator"]["prob_angle_change"])
    prob_timeshift = float(config["DataGenerator"]["prob_timeshift"])
    prob_missing = float(config["DataGenerator"]["prob_missing"])
    prob_dubbling = float(config["DataGenerator"]["prob_dubbling"])
    prob_inaccuracy = float(config["DataGenerator"]["prob_inaccuracy"])
    inaccuracy_wide_x = float(config["DataGenerator"]["inaccuracy_wide_x"])
    inaccuracy_wide_y = float(config["DataGenerator"]["inaccuracy_wide_y"])
    max_height_speed = float(config["DataGenerator"]["max_height_speed"])
    mid_height = float(config["DataGenerator"]["middle_height"])
    count_of_traces = int(config["DataGenerator"]["count_of_traces"])
    np_trajectory = np.array(new_calculation_flight(T, start_time, delta_t, initial_boost, initial_speed, X_start, Y_start, Z_start, initial_angle, max_angle_change, max_angle, max_speed, max_pos_boost, max_neg_boost, min_speed,  prob_boost_change, prob_angle_change, max_height_speed, mid_height))
    np_trajectory = np.array(np_trajectory)
    copy_np_trajectory = copy.deepcopy(np_trajectory)
    new_np_trajectory = inaccuracy_maker(copy_np_trajectory, prob_inaccuracy, inaccuracy_wide_x, inaccuracy_wide_y)
    new_np_trajectory = timeshift_maker(new_np_trajectory, prob_timeshift, delta_t)
    new_np_trajectory = missing_maker(new_np_trajectory, prob_missing)
    new_np_trajectory = dubbling_maker(new_np_trajectory, prob_dubbling)
    new_np_trajectory = slope_zero(new_np_trajectory)
    result_df = pd.DataFrame(new_np_trajectory, columns=['X', 'Y', 'Z','time'])
    result_df['trace'] = 1

    tmp = 0
    for i in range(count_of_traces-1):

      np_trajectory = np.array(new_calculation_flight(T, start_time, delta_t, initial_boost, initial_speed, X_start, Y_start, Z_start, initial_angle, max_angle_change, max_angle, max_speed, max_pos_boost, max_neg_boost, min_speed,  prob_boost_change, prob_angle_change, max_height_speed, mid_height))
      np_trajectory = np.array(np_trajectory)
      copy_np_trajectory = copy.deepcopy(np_trajectory)
      new_np_trajectory = inaccuracy_maker(copy_np_trajectory, prob_inaccuracy, inaccuracy_wide_x, inaccuracy_wide_y)
      new_np_trajectory = timeshift_maker(new_np_trajectory, prob_timeshift, delta_t)
      new_np_trajectory = missing_maker(new_np_trajectory, prob_missing)
      new_np_trajectory = dubbling_maker(new_np_trajectory, prob_dubbling)
      new_np_trajectory = slope_zero(new_np_trajectory)
      add_df = pd.DataFrame(copy.deepcopy(new_np_trajectory), columns=['X', 'Y', 'Z','time'])
      add_df['trace'] = i+2
      result_df=pd.concat([result_df, add_df], ignore_index = True)

    for tmp in range(count_of_traces):
      plt.plot(result_df["X"].where(result_df["trace"]==tmp+1), result_df["Y"].where(result_df["trace"]==tmp+1))

    plt.plot(result_df["X"], result_df["Y"])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, 4500000)
    plt.ylim(-4500000/2, 4500000/2)
    plt.show()

    save_generated_data(result_df)


if __name__ == "__main__":
	main()
