import itertools

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sb
from statannot import add_stat_annotation
import math
import multiprocessing as mp
import os
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional
from sklearn import preprocessing
from isaacgym import gymapi
from pyrr import Quaternion, Vector3
from itertools import chain

from revolve2.core.physics.actor import Actor
from revolve2.core.physics.actor.urdf import to_urdf as physbot_to_urdf
from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    EnvironmentState,
    Runner,
    RunnerState,
)
import math
from random import Random
from typing import List, Optional, Tuple
from pyrr import Quaternion, Vector3
import os
import tempfile
import pickle

from dataclasses import dataclass
from isaacgym import gymapi
from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot
from revolve2.core.modular_robot.brains import *
from revolve2.core.physics.running import ActorControl, Batch, Environment, PosedActor
#from revolve2.runners.isaacgym import LocalRunner
from revolve2.core.physics.actor.urdf import to_urdf as physbot_to_urdf
from revolve2.core.modular_robot.render.render import Render
#from jlo.RL.rl_brain import RLbrain

from tensorforce.environments import Environment as tfenv
from tensorforce.agents import Agent as ag
from tensorforce.execution import Runner as rn
import neat
import visualize

clrs = ['#009900',
        '#EE8610',
        '#7550ff',
        '#876044']

with open(
        '/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/stats_0',
        'rb') as fp:
    statistics = pickle.load(fp)
with open(
        '/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/stats_1',
        'rb') as wp:
    statistics1 = pickle.load(wp)
with open(
        '/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/stats_2',
        'rb') as gp:
    statistics2 = pickle.load(gp)
with open(
        '/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/avg_outputs_0',
        'rb') as fp:
    outputs1 = pickle.load(fp)
with open(
        '/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/avg_outputs_1',
        'rb') as wp:
    outputs2 = pickle.load(wp)
with open(
        '/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/avg_outputs_2',
        'rb') as gp:
    outputs3 = pickle.load(gp)

def load_file(filename):
    with open(filename, 'rb') as gp:
        file = pickle.load(gp)
    return file

s1 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_0/stats_0')
s2 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_1/stats_1')
s3 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_2/stats_2')
s4 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_3/stats_3')
s5 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_4/stats_4')
s6 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_5/stats_5')
s7 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_6/stats_6')
s8 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_7/stats_7')
s9 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_8/stats_8')
s10 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_9/stats_9')
s11 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_0/stats_0')
s12 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_1/stats_1')
s13 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_2/stats_2')
s14 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_3/stats_3')
s15 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_4/stats_4')
s16 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_5/stats_5')
s17 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_6/stats_6')
s18 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_7/stats_7')
s19 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_8/stats_8')
s20= load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_9/stats_9')
s21 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_0/stats_0')
s22 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_1/stats_1')
s23 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_2/stats_2')
s24 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_3/stats_3')
s25 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_4/stats_4')
s26= load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_5/stats_5')
s27 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_6/stats_6')
s28 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_7/stats_7')
s29 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_8/stats_8')
s30 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_9/stats_9')

o1 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_0/avg_outputs_0')
o2 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_1/avg_outputs_1')
o3 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_2/avg_outputs_2')
o4 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_3/avg_outputs_3')
o5 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_4/avg_outputs_4')
o6 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_5/avg_outputs_5')
o7 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_6/avg_outputs_6')
o8 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_7/avg_outputs_7')
o9 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_8/avg_outputs_8')
o10 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_all_10_2/spider_all_9/avg_outputs_9')
o11 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_0/avg_outputs_0')
o12 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_1/avg_outputs_1')
o13 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_2/avg_outputs_2')
o14 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_3/avg_outputs_3')
o15 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_4/avg_outputs_4')
o16 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_5/avg_outputs_5')
o17 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_6/avg_outputs_6')
o18 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_7/avg_outputs_7')
o19 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_8/avg_outputs_8')
o20 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noinner_10_2/spider_noinner_9/avg_outputs_9')
o21 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_0/avg_outputs_0')
o22 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_1/avg_outputs_1')
o23 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_2/avg_outputs_2')
o24 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_3/avg_outputs_3')
o25 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_4/avg_outputs_4')
o26= load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_5/avg_outputs_5')
o27 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_6/avg_outputs_6')
o28 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_7/avg_outputs_7')
o29 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_8/avg_outputs_8')
o30 = load_file('/home/enis/Projects/revolve2/runners/isaacgym/revolve2/runners/isaacgym/neat/spider_noouter_10_2/spider_noouter_9/avg_outputs_9')

dfo1 = pd.DataFrame(outputs1)
dfo2 = pd.DataFrame(outputs2)
dfo3 = pd.DataFrame(outputs3)

df_o1= pd.DataFrame(o1)
df_o2 = pd.DataFrame(o2)
df_o3 = pd.DataFrame(o3)
df_o4 = pd.DataFrame(o4)
df_o5 = pd.DataFrame(o5)
df_o6 = pd.DataFrame(o6)
df_o7 = pd.DataFrame(o7)
df_o8 = pd.DataFrame(o8)
df_o9 = pd.DataFrame(o9)
df_o10 = pd.DataFrame(o10)
df_o11 = pd.DataFrame(o11)
df_o12 = pd.DataFrame(o12)
df_o13 = pd.DataFrame(o13)
df_o14 = pd.DataFrame(o14)
df_o15 = pd.DataFrame(o15)
df_o16 = pd.DataFrame(o16)
df_o17 = pd.DataFrame(o17)
df_o18 = pd.DataFrame(o18)
df_o19 = pd.DataFrame(o19)
df_o20 = pd.DataFrame(o20)
df_o21 = pd.DataFrame(o21)
df_o22 = pd.DataFrame(o22)
df_o23 = pd.DataFrame(o23)
df_o24 = pd.DataFrame(o24)
df_o25 = pd.DataFrame(o25)
df_o26 = pd.DataFrame(o26)
df_o27 = pd.DataFrame(o27)
df_o28 = pd.DataFrame(o28)
df_o29 = pd.DataFrame(o29)
df_o30 = pd.DataFrame(o30)

#print(dfo1,dfo2,dfo3)
st = statistics.generation_statistics
st1 = statistics1.generation_statistics
st2 = statistics2.generation_statistics
#print(st)
def make_dataframe(st):
    vals = []
    for dict in st:
        for inner_dict in dict.values():
            #print(inner_dict.values())
            #vals.append(inner_dict.values())
            vals2 = itertools.chain(inner_dict.values())
            #print(list(vals2))
            vals.append(list(vals2))
    flat_vals = [item for sublist in vals for item in sublist]
    df2 = pd.DataFrame(flat_vals)
    return df2
df2 = make_dataframe(st)
df3 = make_dataframe(st1)
df4 = make_dataframe(st2)

ds1 = make_dataframe(s1.generation_statistics)
ds2 = make_dataframe(s2.generation_statistics)
ds3 = make_dataframe(s3.generation_statistics)
ds4 = make_dataframe(s4.generation_statistics)
ds5 = make_dataframe(s5.generation_statistics)
ds6 = make_dataframe(s6.generation_statistics)
ds7 = make_dataframe(s7.generation_statistics)
ds8 = make_dataframe(s8.generation_statistics)
ds9 = make_dataframe(s9.generation_statistics)
ds10 = make_dataframe(s10.generation_statistics)
ds11 = make_dataframe(s11.generation_statistics)
ds12 = make_dataframe(s12.generation_statistics)
ds13 = make_dataframe(s13.generation_statistics)
ds14 = make_dataframe(s14.generation_statistics)
ds15 = make_dataframe(s15.generation_statistics)
ds16 = make_dataframe(s16.generation_statistics)
ds17 = make_dataframe(s17.generation_statistics)
ds18 = make_dataframe(s18.generation_statistics)
ds19 = make_dataframe(s19.generation_statistics)
ds20 = make_dataframe(s20.generation_statistics)
ds21 = make_dataframe(s21.generation_statistics)
ds22 = make_dataframe(s22.generation_statistics)
ds23 = make_dataframe(s23.generation_statistics)
ds24 = make_dataframe(s24.generation_statistics)
ds25 = make_dataframe(s25.generation_statistics)
ds26 = make_dataframe(s26.generation_statistics)
ds27 = make_dataframe(s27.generation_statistics)
ds28 = make_dataframe(s28.generation_statistics)
ds29 = make_dataframe(s29.generation_statistics)
ds30 = make_dataframe(s30.generation_statistics)

#print(ds1.tail(50))
df_1 = pd.concat((ds1.tail(50), ds2.tail(50),ds3.tail(50),ds4.tail(50),ds5.tail(50),ds6.tail(50),ds7.tail(50),ds8.tail(50),ds9.tail(50),ds10.tail(50)),axis=1).mean()
df_2 = pd.concat((ds11.tail(50), ds12.tail(50),ds13.tail(50),ds14.tail(50),ds15.tail(50),ds16.tail(50),ds17.tail(50),ds18.tail(50),ds19.tail(50),ds20.tail(50)),axis=1).mean()
df_3 = pd.concat((ds21.tail(50), ds22.tail(50),ds23.tail(50),ds24.tail(50),ds25.tail(50),ds26.tail(50),ds27.tail(50),ds28.tail(50),ds29.tail(50),ds30.tail(50)),axis=1).mean()
#print(df_1)

df_ot1 = pd.concat((df_o1.tail(1), df_o2.tail(1),df_o3.tail(1),df_o4.tail(1),df_o5.tail(1),df_o6.tail(1),df_o7.tail(1),df_o8.tail(1),df_o9.tail(1),df_o10.tail(1)),axis=1).mean()
df_ot2 = pd.concat((df_o11.tail(1), df_o12.tail(1),df_o13.tail(1),df_o14.tail(1),df_o15.tail(1),df_o16.tail(1),df_o17.tail(1),df_o18.tail(1),df_o19.tail(1),df_o20.tail(1)),axis=1).mean()
df_ot3 = pd.concat((df_o21.tail(1), df_o22.tail(1),df_o23.tail(1),df_o24.tail(1),df_o25.tail(1),df_o26.tail(1),df_o27.tail(1),df_o28.tail(1),df_o29.tail(1),df_o30.tail(1)),axis=1).mean()

test_arr1 = pd.DataFrame(df_ot1).to_numpy().reshape((8,10))
test_arr2 = pd.DataFrame(df_ot2).to_numpy().reshape((8,10))
test_arr3 = pd.DataFrame(df_ot3).to_numpy().reshape((8,10))

df_ot1 = pd.DataFrame(test_arr1)
df_ot2 = pd.DataFrame(test_arr2)
df_ot3 = pd.DataFrame(test_arr3)

df_out1 = pd.concat((df_ot1.iloc[0],df_ot2.iloc[0],df_ot3.iloc[0]),axis=1)
df_out2 = pd.concat((df_ot1.iloc[1],df_ot2.iloc[1],df_ot3.iloc[1]),axis=1)
df_out3 = pd.concat((df_ot1.iloc[2],df_ot2.iloc[2],df_ot3.iloc[2]),axis=1)
df_out4 = pd.concat((df_ot1.iloc[3],df_ot2.iloc[3],df_ot3.iloc[3]),axis=1)
df_out5 = pd.concat((df_ot1.iloc[4],df_ot2.iloc[4],df_ot3.iloc[4]),axis=1)
df_out6 = pd.concat((df_ot1.iloc[5],df_ot2.iloc[5],df_ot3.iloc[5]),axis=1)
df_out7 = pd.concat((df_ot1.iloc[6],df_ot2.iloc[6],df_ot3.iloc[6]),axis=1)
df_out8 = pd.concat((df_ot1.iloc[7],df_ot2.iloc[7],df_ot3.iloc[7]),axis=1)

df_out1.set_axis(['all', 'no_inner', 'no_outer'], axis='columns', inplace=True)
df_out2.set_axis(['all', 'no_inner', 'no_outer'], axis='columns', inplace=True)
df_out3.set_axis(['all', 'no_inner', 'no_outer'], axis='columns', inplace=True)
df_out4.set_axis(['all', 'no_inner', 'no_outer'], axis='columns', inplace=True)
df_out5.set_axis(['all', 'no_inner', 'no_outer'], axis='columns', inplace=True)
df_out6.set_axis(['all', 'no_inner', 'no_outer'], axis='columns', inplace=True)
df_out7.set_axis(['all', 'no_inner', 'no_outer'], axis='columns', inplace=True)
df_out8.set_axis(['all', 'no_inner', 'no_outer'], axis='columns', inplace=True)



#print(df_out1)



#print(df3)
#print(df4)
#print(flat_vals)
#reshaped_vals = np.asarray(flat_vals).reshape(50,40).T
#print(reshaped_vals)
#df = pd.DataFrame(reshaped_vals)
#print(df)
#print(len(statistics.get_fitness_mean()))
#df.insert(50,'mean',statistics.get_fitness_mean(),True)
#print(df)
#print(df2)
# assign data of lists.
#statistics = {'mean': [0.1, 0.2, 0.3, 0.4], 'generation_index': [0, 1, 2, 3], 'experiment': [1, 2, 3, 4]}
df = pd.concat([df_1,df_2,df_3],axis=1)
df.fillna(0, inplace = True)
df.set_axis(['all', 'no_inner', 'no_outer'], axis='columns', inplace=True)
experiments= [df.loc[:, "all"].values.tolist(),df.loc[:, "no_inner"].values.tolist(),df.loc[:, "no_outer"].values.tolist()]
tests_combinations = [(experiments[i], experiments[j]) \
                          for i in range(len(experiments)) for j in range(i + 1, len(experiments))]

sb.set(rc={"axes.titlesize": 9, "axes.labelsize": 9, 'ytick.labelsize': 9, 'xtick.labelsize': 9})
sb.set_style("whitegrid")
def plot_boxplots_fitness(df):
    #print(len(df.loc[:, "2"].values.tolist()))
    #print(df)
    #print(tests_combinations)
    #y = np.array(np.mean(df2.iloc[:, 0]))
    #print(y)
    #print(pd.melt(df))
    plot = sb.boxplot(data=pd.melt(df), x='variable', y='value',
                      palette=clrs, width=0.4, showmeans=True, linewidth=2, fliersize=6,
                      meanprops={"marker": "o", "markerfacecolor": "yellow", "markersize": "12"})

    plot.tick_params(axis='x', labelrotation=10)
    try:
        if len(tests_combinations) > 0:
            add_stat_annotation(plot,data=pd.melt(df), x='variable', y='value',
                                box_pairs=[('all','no_inner'), ('all','no_outer'),('no_inner','no_outer')],
                                comparisons_correction=None,
                                test='Wilcoxon', text_format='star', fontsize='xx-large', loc='inside',
                                verbose=1)
    except Exception as error:
        print(error)

    # if measures[measure][1] != -math.inf and measures[measure][2] != -math.inf:
    #     plot.set_ylim(measures[measure][1], measures[measure][2])
    plt.xlabel('experiments')
    plt.ylabel('fitness')
    plt.title('Spider')
    #plt.savefig('fitness_boxplots')
    # plot.get_figure().savefig(f'{path}/analysis/{comparison}/box_{measure}_{gen_boxes}.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()
print("\n")

dfo1 = pd.Series(dfo1.values.ravel('F'))
dfo2 = pd.Series(dfo2.values.ravel('F'))
dfo3 = pd.Series(dfo3.values.ravel('F'))

dfo = pd.concat([dfo1,dfo2,dfo3],axis=1)
dfo.fillna(0, inplace=True)
dfo.set_axis(['0', '1', '2'], axis='columns', inplace=True)

def plot_boxplots_outputs(dfo,joint):
    #print(dfo)

    plot = sb.boxplot(data=pd.melt(dfo), x='variable', y='value',
                      palette=clrs, width=0.4, showmeans=True, linewidth=2, fliersize=6,
                      meanprops={"marker": "o", "markerfacecolor": "yellow", "markersize": "12"})

    plot.tick_params(axis='x', labelrotation=10)
    print("\n")
    try:
        if len(tests_combinations) > 0:
            add_stat_annotation(plot,data=pd.melt(dfo), x='variable', y='value',
                                box_pairs=[('all','no_inner'), ('all','no_outer'),('no_inner','no_outer')],
                                comparisons_correction=None,
                                test='Wilcoxon', text_format='star', fontsize='xx-large', loc='inside',
                                verbose=1)
    except Exception as error:
        print(error)

    # if measures[measure][1] != -math.inf and measures[measure][2] != -math.inf:
    #     plot.set_ylim(measures[measure][1], measures[measure][2])
    plt.xlabel('experiments')
    plt.ylabel('outputs')
    plt.title(joint)
    #plt.savefig('outputs_boxplots')
    # plot.get_figure().savefig(f'{path}/analysis/{comparison}/box_{measure}_{gen_boxes}.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()

plot_boxplots_fitness(df)
plot_boxplots_outputs(df_out1,'Joint 0')
plot_boxplots_outputs(df_out2,'Joint 1')
plot_boxplots_outputs(df_out3,'Joint 2')
plot_boxplots_outputs(df_out4,'Joint 3')
plot_boxplots_outputs(df_out5,'Joint 4')
plot_boxplots_outputs(df_out6,'Joint 5')
plot_boxplots_outputs(df_out7,'Joint 6')
plot_boxplots_outputs(df_out8,'Joint 7')
