import torch
import torch.nn as nn
import copy

class PAITracker:
    def __init__(self):
        self.pai_neuron_modules = []
        self.tracked_neuron_modules = []

    def add_pai_neuron_module(self, module):
        self.pai_neuron_modules.append(module)

    def add_tracked_neuron_module(self, module):
        self.tracked_neuron_modules.append(module)

    def clear_all_processors(self):
        for module in self.pai_neuron_modules:
            if hasattr(module, 'clear_processors'):
                module.clear_processors()

# Copyright (c) 2025 Perforated AI
"""PAI configuration file."""

import math
import sys

import torch
import torch.nn as nn

### Global Constants

# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Debug settings
debugging_input_dimensions = 0
confirm_correct_sizes = False

# Confirmation flags for non-recommended options
unwrapped_modules_confirmed = False
weight_decay_accepted = False
checked_skipped_modules = False

# Verbosity settings
verbose = False
extra_verbose = False
silent = False

# Analysis settings
save_old_graph_scores = True

# Testing settings
testing_dendrite_capacity = True

# File format settings
using_safe_tensors = True

# In place for future implementation options of adding multiple candidate
global_candidates = 1

# Graph and visualization settings
drawing_pai = True
test_saves = True
pai_saves = False

# Input dimensions
input_dimensions = [-1, 0, -1, -1]

# Improvement thresholds
improvement_threshold = 0.0001
improvement_threshold_raw = 1e-5

# Weight initialization settings
candidate_weight_initialization_multiplier = 0.01

# SWITCH MODE SETTINGS
DOING_SWITCH_EVERY_TIME = 0
DOING_HISTORY = 1
n_epochs_to_switch = 10
history_lookback = 1
initial_history_after_switches = 0
DOING_FIXED_SWITCH = 2
fixed_switch_num = 250
first_fixed_switch_num = 249
DOING_NO_SWITCH = 3
switch_mode = DOING_HISTORY

# Reset settings
reset_best_score_on_switch = False

# Advanced settings
learn_dendrites_live = False
no_extra_n_modes = True

# Data type
d_type = torch.float

# Dendrite retention settings
retain_all_dendrites = False

# Learning rate management
find_best_lr = True
dont_give_up_unless_learning_rate_lowered = True

# Dendrite attempt settings
max_dendrite_tries = 5
max_dendrites = 100

# Scheduler parameter settings
PARAM_VALS_BY_TOTAL_EPOCH = 0
PARAM_VALS_BY_UPDATE_EPOCH = 1
PARAM_VALS_BY_NEURON_EPOCH_START = 2
param_vals_setting = PARAM_VALS_BY_UPDATE_EPOCH

# Activation function settings
pb_forward_function = torch.sigmoid

### Global Modules

class PAISequential(nn.Sequential):
    def __init__(self, layer_array):
        super(PAISequential, self).__init__()
        self.model = nn.Sequential(*layer_array)
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

### Global objects and variables

# Properly initialize the PAI Tracker
pai_tracker = PAITracker()

# Lists for module types and names to add dendrites to
modules_to_convert = [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear]
module_names_to_convert = ['PAISequential']

modules_to_track = []
module_names_to_track = []
module_ids_to_track = []
modules_to_replace = []
replacement_modules = []
modules_with_processing = []
modules_processing_classes = []
module_names_with_processing = []
module_by_name_processing_classes = []
module_names_to_not_save = ['.base_model']
