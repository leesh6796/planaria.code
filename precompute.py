import pandas
import configparser
import os
import ast
import numpy as np


import copy
import warnings
warnings.filterwarnings('ignore')

import src.benchmarks.benchmarks as benchmarks
from src.simulator.stats import Stats
from src.simulator.simulator import Simulator
from src.sweep.sweep import SimulatorSweep, check_pandas_or_run
from src.utils.utils import *
from src.optimizer.optimizer import optimize_for_order, get_stats_fast

from nn_dataflow import ConvLayer, FCLayer
from nn_dataflow.Layer import DWConvLayer

import pwd
import grp


batch_size = 1

### To load the synthesis results for the systolic subarrays
results_dir = './results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
### The directory where the output csv files will be dumped
numbers_dir = './numbers'
if not os.path.exists(numbers_dir):
    os.makedirs(numbers_dir)


dataframe_list = []


for i in range(16, 17):
    config_file = f'./configs/cmx-{i}.ini'
    verbose = False

    ### Creating a simulator object 
    cmx_sim = Simulator(config_file, verbose)
    cmx_energy_costs = cmx_sim.get_energy_cost()
    print(cmx_sim)

    total_leak_energy, core_dyn_energy, wgt_sram_read_energy, wgt_sram_write_energy, act_sram_read_energy, act_sram_write_energy, out_sram_read_energy, out_sram_write_energy, act_fifo_read_energy, act_fifo_write_energy, out_accum_read_energy, out_accum_write_energy = cmx_energy_costs

    print('*'*50)
    print('Energy costs for Planaria Hardware')
    print('Total leak energy   : {:.3f}'.format(0))
    print('Core dynamic energy : {:.3f} pJ/cycle (for one systolic array core)'.format(core_dyn_energy*1.e3))
    print('Wgt SRAM Read energy    : {:.3f} pJ/bit'.format(wgt_sram_read_energy*1.e3))
    print('Wgt SRAM Write energy   : {:.3f} pJ/bit'.format(wgt_sram_write_energy*1.e3))
    print('Act SRAM Read energy    : {:.3f} pJ/bit'.format(act_sram_read_energy*1.e3))
    print('Act SRAM Write energy   : {:.3f} pJ/bit'.format(act_sram_write_energy*1.e3))
    print('Out SRAM Read energy    : {:.3f} pJ/bit'.format(out_sram_read_energy*1.e3))
    print('Out SRAM Write energy   : {:.3f} pJ/bit'.format(out_sram_write_energy*1.e3))
    print('Act FIFO Read energy   : {:.3f} pJ/bit'.format(act_fifo_read_energy*1.e3))
    print('Act FIFO Write energy   : {:.3f} pJ/bit'.format(act_fifo_write_energy*1.e3))
    print('Out Accumulator Read energy   : {:.3f} pJ/bit'.format(out_accum_read_energy*1.e3))
    print('Out Accumulator Write energy   : {:.3f} pJ/bit'.format(out_accum_write_energy*1.e3))
    print('DRAM Access Energy   : {:.3f} pJ/bit'.format(cmx_sim.cmx.dram_cost))
    print('Wgt Bus Cost   : {:.3f} pJ/bit'.format(cmx_sim.cmx.interconnect_cost['wgt_bus_cost']))
    print('Data Dispatch Bus Cost   : {:.3f} pJ/bit'.format(cmx_sim.cmx.interconnect_cost['data_dispatch_cost']))
    print('*'*50)

    sim_sweep_columns = ['N', 'M',
                    'Number of Threads',
                    'Number of Active Cores',
                    'DRAM Cost (pJ/bit)',
                    'Weight Bus Cost (pJ/bit)',
                    'Data Dispatch Bus Cost (pJ/bit)',
                    'Activation Precision (bits)', 'Weight Precision (bits)',
                    'Network', 'Layer',
                    'Total Cycles', 'Core Cycles', 'Core Compute Cycles', 'Memory wait cycles', 'Energy', 'Data Dispatch Hops', 'Weight Bus Hops',
                    'Weight SRAM Read', 'Weight SRAM Write',
                    'Output SRAM Read', 'Output SRAM Write',
                    'Activation SRAM Read', 'Actiation SRAM Write',
                    'Activation FIFO Read', 'Activation FIFO Write',
                    'Output Acccumulator Read', 'Output Accumulator Write',
                    'DRAM Read', 'DRAM Write',
                    'Tiling', 'Ordering', 'Partitioning', 'Stationary',
                    'Bandwidth (bits/cycle)',
                    'Weight SRAM Size (bits)', 'Output SRAM Size (bits)', 'Input SRAM Size (bits)', 'Act FIFO Size (bits)', 'Out Accumulator Size (Bits)',
                    'Batch size']

    ### Number of Active cores shows how many subarrays are going to be used.
    ### default value: 16, the same as the total number of subarrays.
    ### To get all 16 possibilities, this value needs to be swept from 1 to 16.
    cmx_sim_sweep_csv = os.path.join(numbers_dir, 'planaria-hardware-cmx-{}.csv'.format(cmx_sim.num_active_cores))

    cmx_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)

    ### This function runs all the confuguration possibilities and finds the optimal one for each layer.
    ### At the end it dumps the hardware stats (cycles, energy, number of SRAM/DRAM Accesses)in ./numbers/ directory.
    cmx_results_df = check_pandas_or_run(cmx_sim,
            cmx_sim_sweep_csv, batch_size=batch_size, config_file=config_file)

    data_cmx = pandas.read_csv('./numbers/planaria-hardware-cmx-{}.csv'.format(cmx_sim.num_active_cores))

    final_cmx_col = ['N', 'M', 'Number of Threads', 'Number of Active Cores', 'Network', 'Layer', 'Total Cycles', 'Energy', 'Tiling']
    final_cmx_df = pandas.DataFrame(columns=final_cmx_col)

    for c in final_cmx_col:
        final_cmx_df[c] = data_cmx[c]
    num_tiles_list_cmx = []

    for t in final_cmx_df['Tiling']:
        t_dict = ast.literal_eval(t)
        num_tiles = t_dict['OH/oh'][0] * t_dict['B/b'][0] * t_dict['IC/ic'][0] * t_dict['OC/oc'][0] * t_dict['OW/ow'][0]
        num_tiles_list_cmx.append(num_tiles)
    final_cmx_df['Tiling'] = num_tiles_list_cmx
    
    final_cmx_df.to_csv('./numbers/planaria-hardware-summary-{}.csv'.format(cmx_sim.num_active_cores))