# -*- coding: utf-8 -*-
"""
[20180320 Treatment Effects with RNNs] propensity_weight_generation_script
Created on 22/4/2018 1:54 PM

Script to generate propensity weights

@author: limsi
"""

import configs

import core_routines as core
from core_routines import test

import numpy as np
import logging
import os

import tensorflow as tf

import seaborn as sns
sns.set()

ROOT_FOLDER = configs.ROOT_FOLDER
MODEL_ROOT = configs.MODEL_ROOT

expt_name = "treatment_effects"

# EDIT ME ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Specify parameters for model to load - optimal set from paper listed
action_inputs_only = configs.load_optimal_parameters('treatment_rnn_action_inputs_only',
                                                     expt_name,
                                                     add_net_name=True)
action_w_trajectory_inputs = configs.load_optimal_parameters('treatment_rnn',
                                                             expt_name,
                                                             add_net_name=True)
censor_w_action_inputs_only = configs.load_optimal_parameters('censor_rnn_action_inputs_only',
                                                              expt_name,
                                                              add_net_name=True)
censor_w_trajectory_inputs = configs.load_optimal_parameters('censor_rnn',
                                                             expt_name,
                                                             add_net_name=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    # Generate propensity weights for validation data as well - used for MSM which is calibrated on train + valid data
    b_with_validation = False
    # Generate non-stabilised IPTWs (default false)
    b_denominator_only = False

    # Setup tensorflow - setup session to use cpu/gpu
    tf_device = 'cpu'
    if tf_device == "cpu":
        tf_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})
    else:
        tf_config= tf.ConfigProto(log_device_placement=False, device_count={'GPU': 1})
        tf_config.gpu_options.allow_growth = True

    # Config + activation functions
    activation_map = {'rnn_propensity_weighted': ("elu", 'linear'),
                      'rnn_model': ("elu", 'linear'),
                      'rnn_model_bptt': ("elu", 'linear'),
                      'treatment_rnn': ("tanh", 'sigmoid'),
                      'treatment_rnn_action_inputs_only': ("tanh", 'sigmoid'),
                      'treatment_rnn_softmax': ("tanh", 'sigmoid'),
                      'treatment_rnn_action_inputs_only_softmax': ("tanh", 'sigmoid'),
                      'censor_rnn': ("tanh", 'sigmoid'),
                      'censor_rnn_action_inputs_only': ("tanh", 'sigmoid')
                      }

    configs = {'action_num': action_inputs_only,
               'action_den': action_w_trajectory_inputs,
               'censor_num': censor_w_action_inputs_only,
               'censor_den': censor_w_trajectory_inputs}

    # Setup the simulated datasets
    chemo_coeff = 10
    radio_coeff = 10
    b_load = True
    b_save = True
    pickle_map = core.get_cancer_sim_data(chemo_coeff, radio_coeff, b_load=b_load, b_save=b_save)

    chemo_coeff = pickle_map['chemo_coeff']
    radio_coeff = pickle_map['radio_coeff']
    num_time_steps = pickle_map['num_time_steps']
    training_data = pickle_map['training_data']
    validation_data = pickle_map['validation_data']
    test_data = pickle_map['test_data']
    scaling_data = pickle_map['scaling_data']

    # Generate propensity weights for validation data if required
    if b_with_validation:
        for k in training_data:
            training_data[k] = np.concatenate([training_data[k], validation_data[k]])

    ##############################################################################################################
    # Functions
    def get_predictions(config):

        net_name = config[0]

        hidden_activation, output_activation = activation_map[net_name]

        # Pull datasets
        b_predict_actions = "treatment_rnn" in net_name
        b_use_actions_only = "rnn_action_inputs_only" in net_name
        b_predict_censoring = "censor_rnn" in net_name

        # Extract only relevant trajs and shift data
        training_processed = core.get_processed_data(training_data, scaling_data, b_predict_actions, b_use_actions_only,
                                                     b_predict_censoring)
        validation_processed = core.get_processed_data(validation_data, scaling_data, b_predict_actions,
                                                       b_use_actions_only, b_predict_censoring)

        num_features = training_processed['scaled_inputs'].shape[-1]  # 4 if not b_use_actions_only else 3
        num_outputs = training_processed['scaled_outputs'].shape[-1]  # 1 if not b_predict_actions else 3  # 5

        # Unpack remaining variables
        dropout_rate = config[1]
        memory_multiplier = config[2] / num_features
        num_epochs = config[3]
        minibatch_size = config[4]
        learning_rate = config[5]
        max_norm = config[6]



        model_folder = os.path.join(MODEL_ROOT, net_name)
        means, outputs, _, _ = test(training_processed, validation_processed, training_processed, tf_config,
                                     net_name, expt_name, dropout_rate, num_features, num_outputs,
                                     memory_multiplier, num_epochs, minibatch_size, learning_rate, max_norm,
                                     hidden_activation, output_activation, model_folder)

        return means, outputs

    def get_weights(probs, targets, b_predict_censoring):

        if b_predict_censoring:

            # Assume the last state is the 'ok' state, take only that while preseving 3d arr
            return probs[:, :, -1]
        else:
            w = probs*targets + (1-probs) * (1-targets)

            return w.prod(axis=2)

    def get_weights_from_config(config):
        net_name = config[0]
        b_predict_censoring = "censor_rnn" in net_name

        probs, targets = get_predictions(config)

        return get_weights(probs, targets, b_predict_censoring)

    def get_probabilities_from_config(config):
        net_name = config[0]
        b_predict_censoring = "censor_rnn" in net_name

        probs, targets = get_predictions(config)

        return probs


    ##############################################################################################################

    # Action with trajs
    weights = {k: get_weights_from_config(configs[k]) for k in configs}

    den = weights['action_den'] * weights['censor_den']
    num = weights['action_num'] * weights['censor_num']

    propensity_weights = 1.0/den if b_denominator_only else num/den

    # truncation @ 95th and 5th percentiles
    UB = np.percentile(propensity_weights, 99)
    LB = np.percentile(propensity_weights, 1)

    propensity_weights[propensity_weights > UB] = UB
    propensity_weights[propensity_weights < LB] = LB

    # Adjust so for 3 trajectories here
    horizon = 1
    (num_patients, num_time_steps) = propensity_weights.shape
    output = np.ones((num_patients, num_time_steps, horizon))

    tmp = np.ones((num_patients, num_time_steps))
    tmp[:, 1:] = propensity_weights[:, :-1]
    propensity_weights = tmp

    for i in range(horizon):
        output[:, :num_time_steps-i, i] = propensity_weights[:, i:]

    propensity_weights = output.cumprod(axis=2)

    suffix = "" if not b_denominator_only else "_den_only"

    if b_with_validation:
        save_file = os.path.join(MODEL_ROOT, "propensity_scores_w_validation{}".format(suffix))
    else:
        save_file = os.path.join(MODEL_ROOT, "propensity_scores{}".format(suffix))

    print("Saving propensity weights to ", save_file)
    np.save(save_file, propensity_weights)

    print("Weights saved!")

