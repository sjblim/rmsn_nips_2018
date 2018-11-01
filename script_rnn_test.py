# -*- coding: utf-8 -*
"""
[20180320 Treatment Effects with RNNs] test_script
Created on 22/3/2018 5:48 PM

@author: limsi
"""

import configs

from core_routines import test
import core_routines as core

from libs.model_rnn import RnnModel
import libs.net_helpers as helpers

from sklearn.metrics import roc_auc_score, average_precision_score

import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import os

import matplotlib.pyplot as plt
import tensorflow as tf

import seaborn as sns
sns.set()

ROOT_FOLDER = configs.ROOT_FOLDER
MODEL_ROOT = configs.MODEL_ROOT
RESULTS_FOLDER = configs.RESULTS_FOLDER

# Default params:
expt_name = "treatment_effects"

# EDIT ME! ######################################################################################
# Optimal network parameters to load for testing!
configs = [
configs.load_optimal_parameters('rnn_propensity_weighted',
                                 expt_name,
                                 add_net_name=True)
          ]

##################################################################################################


if __name__ == "__main__":

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    # Setup tensorflow
    tf_device = 'gpu'
    if tf_device == "cpu":
        tf_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})
    else:
        tf_config= tf.ConfigProto(log_device_placement=False, device_count={'GPU': 1})
        tf_config.gpu_options.allow_growth = True

    # Config
    activation_map = {'rnn_propensity_weighted': ("elu", 'linear'),
                      'rnn_propensity_weighted_binary': ("elu", 'linear'),
                      'rnn_propensity_weighted_logistic': ("elu", 'linear'),
                      'rnn_model': ("elu", 'linear'),
                      'treatment_rnn': ("tanh", 'sigmoid'),
                      'treatment_rnn_actions_only': ("tanh", 'sigmoid')}
    max_coeff = 10
    window_sizes = [15]  # Use the default 15 day lookback window
    projection_map = {}
    mse_by_followup = {}
    for config in configs:

        net_name = config[0]

        projection_map[net_name] = {}

        # pick out scaling data first
        main_map = core.get_cancer_sim_data(10, 10, b_load=True, b_save=True, seed=100)
        scaling_data = main_map['scaling_data']

        for window_size in window_sizes:
            for chemo_coeff in [i for i in range(max_coeff + 1)]:
                for radio_coeff in [i for i in range(max_coeff + 1)]:

                    logging.info("Processing {} chemo_coeff={}, radio_coeff={}".format(net_name, chemo_coeff, radio_coeff))

                    seed = 100 if chemo_coeff == max_coeff and radio_coeff == max_coeff and window_size == 15 else int(100 * np.random.rand())

                    pickle_map = core.get_cancer_sim_data(chemo_coeff, radio_coeff, b_load=True, b_save=True, seed=seed,
                                                          window_size=window_size)

                    chemo_coeff = pickle_map['chemo_coeff']
                    radio_coeff = pickle_map['radio_coeff']
                    num_time_steps = pickle_map['num_time_steps']
                    training_data = pickle_map['training_data']
                    validation_data = pickle_map['validation_data']
                    test_data = pickle_map['test_data']

                    # scaling_data = pickle_map['scaling_data']  # use scaling data from above

                    # Setup some params
                    b_predict_actions = "treatment_rnn" in net_name
                    b_propensity_weight = "rnn_propensity_weighted" in net_name
                    b_use_actions_only = "treatment_rnn_action_inputs_only" in net_name
                    b_predict_censoring = 'censor' in net_name

        # In[*]: Compute base MSEs
                    # Extract only relevant trajs and shift data
                    training_processed = core.get_processed_data(training_data, scaling_data, b_predict_actions,
                                                                 b_use_actions_only,  b_predict_censoring)
                    validation_processed = core.get_processed_data(validation_data, scaling_data, b_predict_actions,
                                                                   b_use_actions_only,  b_predict_censoring)
                    test_processed = core.get_processed_data(test_data, scaling_data, b_predict_actions,
                                                             b_use_actions_only, b_predict_censoring)

                    num_features = training_processed['scaled_inputs'].shape[-1]  # 4 if not b_use_actions_only else 3
                    num_outputs = training_processed['scaled_outputs'].shape[-1]  # 1 if not b_predict_actions else 3  # 5

                    # Pull remaining params
                    dropout_rate = config[1]
                    memory_multiplier = config[2] / num_features
                    num_epochs = config[3]
                    minibatch_size = config[4]
                    learning_rate = config[5]
                    max_norm = config[6]
                    backprop_length = 60  # we've fixed this
                    hidden_activation = activation_map[net_name][0]
                    output_activation = activation_map[net_name][1]

                    # Run tests
                    model_folder = os.path.join(MODEL_ROOT, net_name)

                    means, output, mse, test_states \
                        = test(training_processed, validation_processed, test_processed, tf_config,
                               net_name, expt_name, dropout_rate, num_features, num_outputs,
                               memory_multiplier, num_epochs, minibatch_size, learning_rate, max_norm,
                               hidden_activation, output_activation, model_folder,
                               b_use_state_initialisation=False, b_dump_all_states=True)

                    active_entries = test_processed['active_entries']

                    def get_mse_at_follow_up_time(mean, output, active_entires):
                        mses = np.sum(np.sum((mean - output) **2 *active_entires, axis=-1), axis=0) \
                               / active_entires.sum(axis=0).sum(axis=-1)

                        return pd.Series(mses, index=[idx for idx in range(len(mses))], name=net_name)

                    # Add results over time
                    if chemo_coeff == 0 and radio_coeff == 0:
                        if window_size not in mse_by_followup:
                            mse_by_followup[window_size] = pd.DataFrame()

                        mse_by_followup[window_size][net_name] = get_mse_at_follow_up_time(means, output, active_entries)

                    # Add stnd mses
                    if window_size not in projection_map[net_name]:
                        projection_map[net_name][window_size] =\
                            pd.DataFrame([], index=[i for i in range(max_coeff + 1)],
                                             columns=[i for i in range(max_coeff + 1)])

                    projection_map[net_name][window_size][chemo_coeff][radio_coeff] = mse


    # In[*]: Save outputs

    #for win in mse_by_followup:
    #    mse_by_followup[win].to_csv(os.path.join(RESULTS_FOLDER, "mse_by_followup_rnns" + str(win) + "_mse.csv"))

    for k in projection_map:
        for i in projection_map[k]:
            #projection_map[k][i].to_csv(os.path.join(RESULTS_FOLDER, k + "_one_step_action_window-" + str(i) + "_mse.csv"))
            projection_map[k][i].to_csv(
                os.path.join(RESULTS_FOLDER, k + "_1_mse.csv"))
