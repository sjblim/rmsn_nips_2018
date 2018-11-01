# -*- coding: utf-8 -*-
"""
[20180320 Treatment Effects with RNNs] sim_seq2seq_test
Created on 6/5/2018 11:22 AM

@author: limsi
"""

import configs

import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import os

from core_routines import test
import core_routines as core
from script_decoder_fit import process_seq_data

ROOT_FOLDER = configs.ROOT_FOLDER
MODEL_ROOT = configs.MODEL_ROOT
RESULTS_FOLDER = configs.RESULTS_FOLDER

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

expt_name = "treatment_effects"

# EDIT ME! ######################################################################################
# Which networks to load for testing
decoder_specifications = {
        'rnn_propensity_weighted_seq2seq': configs.load_optimal_parameters('rnn_propensity_weighted_seq2seq',
                                                                           expt_name)
    }

encoder_specifications = {
    'rnn_propensity_weighted': configs.load_optimal_parameters('rnn_propensity_weighted',
                                                               expt_name)
}

net_names = ['rnn_propensity_weighted']
##################################################################################################


# In[*]: Main routine
if __name__ == "__main__":

    logging.info("Running hyperparameter optimisation")

    # Setup params for datas
    tf_device = 'gpu'
    b_apply_memory_adapter = True
    b_single_layer = True  # single or multilayer memory adapter
    max_coeff = 10

    activation_map = {'rnn_propensity_weighted': ("elu", 'linear'),
                      'rnn_propensity_weighted_den_only': ("elu", 'linear'),
                      'rnn_propensity_weighted_logistic': ("elu", 'linear'),
                      'rnn_model': ("elu", 'linear')
                      }

    # Setup tensorflow
    if tf_device == "cpu":
        tf_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})
    else:
        tf_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 1})
        tf_config.gpu_options.allow_growth = True

    # In[*]: Config specific functions
    def generate_seq2seq_data(net_name, chemo_coeff, radio_coeff, window_size=15):

        # Setup the simulated datasets
        b_load = True
        b_save = True
        pickle_map = core.get_cancer_sim_data(chemo_coeff, radio_coeff, b_load=b_load, b_save=b_save, window_size=window_size)

        chemo_coeff = pickle_map['chemo_coeff']
        radio_coeff = pickle_map['radio_coeff']
        num_time_steps = pickle_map['num_time_steps']
        training_data = pickle_map['training_data']
        validation_data = pickle_map['validation_data']
        test_data = pickle_map['test_data']

        # Use scaling data only from the original
        scale_map = core.get_cancer_sim_data(10, 10, b_load=True, b_save=True, seed=100)
        scaling_data = scale_map['scaling_data']

        # Pull datasets
        b_predict_actions = "treatment_rnn" in net_name
        b_predict_censoring = "censor_rnn" in net_name
        b_propensity_weight = "rnn_propensity_weighted" in net_name
        b_use_actions_only = "rnn_action_inputs_only" in net_name

        # checks
        if b_predict_actions and b_predict_censoring:
            raise ValueError("problem with RNN! RNN is both actions and censoring")

        # Extract only relevant trajs and shift data
        training_processed = core.get_processed_data(training_data, scaling_data, b_predict_actions, b_use_actions_only,
                                                     b_predict_censoring)
        validation_processed = core.get_processed_data(validation_data, scaling_data, b_predict_actions,
                                                       b_use_actions_only,
                                                       b_predict_censoring)
        test_processed = core.get_processed_data(test_data, scaling_data, b_predict_actions, b_use_actions_only,
                                                 b_predict_censoring)

        num_features = training_processed['scaled_inputs'].shape[-1]  # 4 if not b_use_actions_only else 3
        num_outputs = training_processed['scaled_outputs'].shape[-1]  # 1 if not b_predict_actions else 3  # 5

        # Load propensity weights if they exist
        if b_propensity_weight:
            # raise NotImplementedError("Propensity weights will be added later")

            if net_name == 'rnn_propensity_weighted_den_only':
                propensity_weights = np.load(os.path.join(MODEL_ROOT, "propensity_scores_den_only.npy"))
            elif net_name == "rnn_propensity_weighted_logistic":
                propensity_weights = np.load(os.path.join(MODEL_ROOT, "propensity_scores.npy"))
                tmp = np.load(os.path.join(MODEL_ROOT, "propensity_scores_logistic.npy"))
                propensity_weights = tmp[:propensity_weights.shape[0], :, :]
            else:
                propensity_weights = np.load(os.path.join(MODEL_ROOT, "propensity_scores.npy"))

            training_processed['propensity_weights'] = propensity_weights

        logging.info("Loading basic network to generate states: {}".format(net_name))

        if net_name not in encoder_specifications:
            raise ValueError("Can't find term in hyperparameter specifications")

        spec = encoder_specifications[net_name]
        logging.info("Using specifications for {}: {}".format(net_name, spec))
        dropout_rate = spec[0]
        memory_multiplier = spec[1]/num_features
        num_epochs = spec[2]
        minibatch_size = spec[3]
        learning_rate = spec[4]
        max_norm = spec[5]

        hidden_activation, output_activation = activation_map[net_name]

        model_folder = os.path.join(MODEL_ROOT, net_name)
        train_preds, _, _, train_states = test(training_processed, validation_processed, training_processed, tf_config,
                                               net_name, expt_name, dropout_rate, num_features, num_outputs,
                                               memory_multiplier, num_epochs, minibatch_size, learning_rate, max_norm,
                                               hidden_activation, output_activation, model_folder,
                                               b_use_state_initialisation=False, b_dump_all_states=True)

        valid_preds, _, _, valid_states = test(training_processed, validation_processed, validation_processed,
                                               tf_config,
                                               net_name, expt_name, dropout_rate, num_features, num_outputs,
                                               memory_multiplier, num_epochs, minibatch_size, learning_rate, max_norm,
                                               hidden_activation, output_activation, model_folder,
                                               b_use_state_initialisation=False, b_dump_all_states=True)

        test_preds, _, _, test_states = test(training_processed, validation_processed, test_processed, tf_config,
                                             net_name, expt_name, dropout_rate, num_features, num_outputs,
                                             memory_multiplier, num_epochs, minibatch_size, learning_rate, max_norm,
                                             hidden_activation, output_activation, model_folder,
                                             b_use_state_initialisation=False, b_dump_all_states=True)


        # Repackage inputs here
        training_processed = process_seq_data(training_processed, train_states,
                                              num_features_to_include=num_outputs)
        validation_processed = process_seq_data(validation_processed, valid_states,
                                                num_features_to_include=num_outputs)
        test_processed = process_seq_data(test_processed, test_states,
                                          num_features_to_include=num_outputs)

        return training_processed, validation_processed, test_processed

    # In[*] Start Running testing procedure
    results_map = {}
    projection_map = {}
    for net_name in net_names:

        # Test
        suffix = "_seq2seq"
        suffix += "_no_adapter" if not b_apply_memory_adapter else ""
        suffix += "_multi_layer" if not b_single_layer else ""


        seq_net_name = net_name + suffix
        model_folder = os.path.join(MODEL_ROOT, seq_net_name)

        if seq_net_name not in decoder_specifications:
            raise ValueError("Cannot find decoder specifications for {}".format(seq_net_name))

        results_map[seq_net_name] = pd.DataFrame([], index=[i for i in range(max_coeff + 1)],
                                             columns=[i for i in range(max_coeff + 1)])
        projection_map[seq_net_name] = {}

        for chemo_coeff in [i for i in range(max_coeff + 1)]:
            for radio_coeff in [i for i in range(max_coeff + 1)]:

                    # Data setup
                    training_processed, validation_processed, test_processed = \
                                                            generate_seq2seq_data(net_name, chemo_coeff, radio_coeff)

                    num_features = training_processed['scaled_inputs'].shape[-1]
                    num_outputs = training_processed['scaled_outputs'].shape[-1]

                    # Pulling specs
                    spec = decoder_specifications[seq_net_name]
                    logging.info("Using specifications for {}: {}".format(seq_net_name, spec))
                    dropout_rate = spec[0]
                    memory_multiplier = spec[1]/num_features  # hack to recover correct size
                    num_epochs = spec[2]
                    minibatch_size = spec[3]
                    learning_rate = spec[4]
                    max_norm = spec[5]
                    hidden_activation, output_activation = activation_map[net_name]

                    _, _, mse, _ \
                        = test(training_processed, validation_processed, test_processed, tf_config,
                               seq_net_name, expt_name, dropout_rate, num_features, num_outputs,
                               memory_multiplier, num_epochs, minibatch_size, learning_rate, max_norm,
                               hidden_activation, output_activation, model_folder,
                               b_use_state_initialisation=True, b_dump_all_states=False,
                               b_mse_by_time=True,
                               b_use_memory_adapter=b_apply_memory_adapter)

                    mse = mse.flatten()

                    for proj_idx in range(mse.shape[0]):

                        if proj_idx > 4:
                            break

                        if proj_idx not in projection_map[seq_net_name]:
                            projection_map[seq_net_name][proj_idx] = \
                                pd.DataFrame([], index=[i for i in range(max_coeff + 1)],
                                             columns=[i for i in range(max_coeff + 1)])

                        projection_map[seq_net_name][proj_idx][chemo_coeff][radio_coeff] = mse[proj_idx]


    # In[*]: Save results

    #for k in results_map:
    #    results_map[k].to_csv(os.path.join(RESULTS_FOLDER, k+"_mse.csv"))

    for k in projection_map:
        for i in projection_map[k]:
            projection_map[k][i].to_csv(os.path.join(RESULTS_FOLDER, k + "_" + str(i+2) + "_mse.csv"))


