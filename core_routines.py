# -*- coding: utf-8 -*-

"""
Treatment Effects with RNNs:

Common routines to use across all training scripts

Created on 30/4/2018 10:08 PM
@author: Bryan
"""

import configs

from libs.model_rnn import RnnModel
import libs.net_helpers as helpers
from simulation import cancer_simulation as sim

import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import os
import pickle

ROOT_FOLDER = configs.ROOT_FOLDER
MODEL_ROOT = configs.MODEL_ROOT

#--------------------------------------------------------------------------
# Training routine
#--------------------------------------------------------------------------
def train(net_name,
          expt_name,
          training_dataset, validation_dataset, test_dataset,
          dropout_rate,
          memory_multiplier,
          num_epochs,
          minibatch_size,
          learning_rate,
          max_norm,
          use_truncated_bptt,
          num_features,
          num_outputs,
          model_folder,
          hidden_activation,
          output_activation,
          tf_config,
          additonal_info="",
          b_use_state_initialisation=False,
          b_use_seq2seq_feedback=False,
          b_use_seq2seq_training_mode=False,
          adapter_multiplier=0,
          b_use_memory_adapter=False):

    """
    Common training routine to all RNN models - seq2seq + standard
    """

    min_epochs = 1

    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

        tf_data_train = convert_to_tf_dataset(training_dataset)
        tf_data_valid = convert_to_tf_dataset(validation_dataset)
        tf_data_test = convert_to_tf_dataset(test_dataset)

        # Setup default hidden layer size
        hidden_layer_size = int(memory_multiplier * num_features)

        if b_use_state_initialisation:

            full_state_size = int(training_dataset['initial_states'].shape[-1])

            adapter_size = adapter_multiplier * full_state_size

        else:
            adapter_size = 0

            # Training simulation
        model_parameters = {'net_name': net_name,
                            'experiment_name': expt_name,
                            'training_dataset': tf_data_train,
                            'validation_dataset': tf_data_valid,
                            'test_dataset': tf_data_test,
                            'dropout_rate': dropout_rate,
                            'input_size': num_features,
                            'output_size': num_outputs,
                            'hidden_layer_size': hidden_layer_size,
                            'num_epochs': num_epochs,
                            'minibatch_size': minibatch_size,
                            'learning_rate': learning_rate,
                            'max_norm': max_norm,
                            'model_folder': model_folder,
                            'hidden_activation': hidden_activation,
                            'output_activation': output_activation,
                            'backprop_length': 60,  # backprop over 60 timesteps for truncated backpropagation through time
                            'softmax_size': 0, #not used in this paper, but allows for categorical actions
                            'performance_metric': 'xentropy' if output_activation == 'sigmoid' else 'mse',
                            'use_seq2seq_feedback': b_use_seq2seq_feedback,
                            'use_seq2seq_training_mode': b_use_seq2seq_training_mode,
                            'use_memory_adapter': b_use_memory_adapter,
                            'memory_adapter_size': adapter_size}

        # Get the right model
        model = RnnModel(model_parameters)
        serialisation_name = model.serialisation_name

        if helpers.hyperparameter_result_exists(model_folder, net_name, serialisation_name):
            logging.warning("Combination found: skipping {}".format(serialisation_name))
            return helpers.load_hyperparameter_results(model_folder, net_name)

        training_handles = model.get_training_graph(use_truncated_bptt=use_truncated_bptt,
                                                    b_use_state_initialisation=b_use_state_initialisation)
        validation_handles = model.get_prediction_graph(use_validation_set=True, with_dropout=False,
                                                        b_use_state_initialisation=b_use_state_initialisation)

        # Start optimising
        num_minibatches = int(np.ceil(training_dataset['scaled_inputs'].shape[0] / model_parameters['minibatch_size']))

        i = 1
        epoch_count = 1
        step_count = 1
        min_loss = np.inf
        with sess.as_default():

            sess.run(tf.global_variables_initializer())

            optimisation_summary = pd.Series([])

            while True:
                try:
                    loss, _ = sess.run([training_handles['loss'],
                                        training_handles['optimiser']])

                    # Flog output
                    logging.info("Epoch {} | iteration = {} of {}, loss = {} | net = {} | info = {}".format(
                        epoch_count,
                        step_count,
                        num_minibatches,
                        loss,
                        model.net_name,
                        additonal_info))

                    if step_count == num_minibatches:

                        # Reinit dataset
                        sess.run(validation_handles['initializer'])

                        means = []
                        UBs = []
                        LBs = []
                        while True:
                            try:
                                mean, upper_bound, lower_bound = sess.run([validation_handles['mean'],
                                                                           validation_handles['upper_bound'],
                                                                           validation_handles['lower_bound']])

                                means.append(mean)
                                UBs.append(upper_bound)
                                LBs.append(lower_bound)
                            except tf.errors.OutOfRangeError:
                                break

                        means = np.concatenate(means, axis=0)*training_dataset['output_stds'] \
                                + training_dataset['output_means']
                        UBs = np.concatenate(UBs, axis=0)*training_dataset['output_stds'] \
                              + training_dataset['output_means']
                        LBs = np.concatenate(LBs, axis=0)*training_dataset['output_stds'] \
                              + training_dataset['output_means']

                        active_entries = validation_dataset['active_entries']
                        output = validation_dataset['outputs']

                        if model_parameters['performance_metric'] == "mse":
                            validation_loss = np.sum((means - output)**2 * active_entries) / np.sum(active_entries)

                        elif model_parameters['performance_metric'] == "xentropy":
                            _, _,features_size = output.shape
                            partition_idx = features_size

                            # Do binary first
                            validation_loss = np.sum((output[:, :, :partition_idx] * -np.log(means[:, :, :partition_idx] + 1e-8)
                                                     + (1 - output[:, :, :partition_idx]) * -np.log(1 - means[:, :, :partition_idx] + 1e-8))
                                                     * active_entries[:, :, :partition_idx]) \
                                              / np.sum(active_entries[:, :, :partition_idx])

                        optimisation_summary[epoch_count] = validation_loss

                        # Compute validation loss
                        logging.info("Epoch {} Summary| Validation loss = {} | net = {} | info = {}".format(
                            epoch_count,
                            validation_loss,
                            model.net_name,
                            additonal_info))

                        if np.isnan(validation_loss):
                            logging.warning("NAN Loss found, terminating routine")
                            break

                        # Save model and loss trajectories
                        if validation_loss < min_loss and epoch_count > min_epochs:
                            cp_name = serialisation_name + "_optimal"
                            helpers.save_network(sess, model_folder, cp_name, optimisation_summary)
                            min_loss = validation_loss

                        # Update
                        epoch_count += 1
                        step_count = 0

                    step_count += 1
                    i += 1

                except tf.errors.OutOfRangeError:
                    break

            # Save final
            cp_name = serialisation_name + "_final"
            helpers.save_network(sess, model_folder, cp_name, optimisation_summary)
            helpers.add_hyperparameter_results(optimisation_summary, model_folder, net_name, serialisation_name)

            hyperparam_df = helpers.load_hyperparameter_results(model_folder, net_name)

            logging.info("Terminated at iteration {}".format(i))
            sess.close()

    return hyperparam_df

#--------------------------------------------------------------------------
# Test routine
#--------------------------------------------------------------------------
def test(training_dataset,
         validation_dataset,
         test_dataset,
         tf_config,
         net_name,
         expt_name,
         dropout_rate,
         num_features,
         num_outputs,
         memory_multiplier,
         num_epochs,
         minibatch_size,
         learning_rate,
         max_norm,
         hidden_activation,
         output_activation,
         model_folder,
         b_use_state_initialisation=False,
         b_dump_all_states=False,
         b_mse_by_time=False,
         b_use_seq2seq_feedback=False,
         b_use_seq2seq_training_mode=False,
         adapter_multiplier=0,
         b_use_memory_adapter=False
         ):

    """
    Common test routine to all RNN models - seq2seq + standard
    """

    # Start with graph
    tf.reset_default_graph()

    with tf.Session(config=tf_config) as sess:
        tf_data_train = convert_to_tf_dataset(training_dataset)
        tf_data_valid = convert_to_tf_dataset(validation_dataset)
        tf_data_test = convert_to_tf_dataset(test_dataset)

        # For decoder training with external state inputs
        if b_use_state_initialisation:

            full_state_size = int(training_dataset['initial_states'].shape[-1])

            adapter_size = adapter_multiplier * full_state_size

        else:
            adapter_size = 0

        # Training simulation
        model_parameters = {'net_name': net_name,
                            'experiment_name': expt_name,
                            'training_dataset': tf_data_train,
                            'validation_dataset': tf_data_valid,
                            'test_dataset': tf_data_test,
                            'dropout_rate': dropout_rate,
                            'input_size': num_features,
                            'output_size': num_outputs,
                            'hidden_layer_size': int(memory_multiplier * num_features),
                            'num_epochs': num_epochs,
                            'minibatch_size': minibatch_size,
                            'learning_rate': learning_rate,
                            'max_norm': max_norm,
                            'model_folder': model_folder,
                            'hidden_activation': hidden_activation,
                            'output_activation': output_activation,
                            'backprop_length': 60,  # Length for truncated backpropagation over time, matches max time steps here.
                            'softmax_size': 0, #not used in this paper, but allows for categorical actions
                            'performance_metric': 'xentropy' if output_activation == 'sigmoid' else 'mse',
                            'use_seq2seq_feedback': b_use_seq2seq_feedback,
                            'use_seq2seq_training_mode': b_use_seq2seq_training_mode,
                            'use_memory_adapter': b_use_memory_adapter,
                            'memory_adapter_size': adapter_size}


        # Start optimising
        with sess.as_default():

            sess.run(tf.global_variables_initializer())

            # Get the right model
            model = RnnModel(model_parameters)
            handles = model.get_prediction_graph(use_validation_set=False if 'treatment_rnn' not in net_name  else None,
                                                 with_dropout=False,
                                                 b_use_state_initialisation=b_use_state_initialisation,
                                                 b_dump_all_states=b_dump_all_states)

            # Load checkpoint
            serialisation_name = model.serialisation_name
            cp_name = serialisation_name + "_optimal"
            _ = helpers.load_network(sess, model_folder, cp_name)

            # Init
            sess.run(handles['initializer'])

            # Get all the data out in chunks
            means = []
            UBs = []
            LBs = []
            states =[]
            while True:
                try:
                    mean, upper_bound, lower_bound, ave_states \
                        = sess.run([handles['mean'],
                                    handles['upper_bound'],
                                    handles['lower_bound'],
                                    handles['ave_states']])

                    means.append(mean)
                    UBs.append(upper_bound)
                    LBs.append(lower_bound)
                    states.append(ave_states)
                except tf.errors.OutOfRangeError:
                    break

            means = np.concatenate(means, axis=0) * training_dataset['output_stds']\
                    + training_dataset['output_means']
            UBs = np.concatenate(UBs, axis=0) * training_dataset['output_stds'] \
                  + training_dataset['output_means']
            LBs = np.concatenate(LBs, axis=0) * training_dataset['output_stds'] \
                  + training_dataset['output_means']
            states = np.concatenate(states, axis=0)

            active_entries = test_dataset['active_entries'] \
                if net_name != 'treatment_rnn' else training_dataset['active_entries']
            output = test_dataset['outputs'] \
                if net_name != 'treatment_rnn' else training_dataset['outputs']

            # prediction_map[net_name] = means
            # output_map[net_name] = output

            if b_mse_by_time:
                mse = np.sum((means - output) ** 2 * active_entries, axis=0) / np.sum(active_entries, axis=0)
            else:
                mse = np.sum((means - output) ** 2 * active_entries) / np.sum(active_entries)

            # results[net_name] = mse
            # print(net_name, mse)
            sess.close()

        return means, output, mse, states

#--------------------------------------------------------------------------
# Data processing functions
#--------------------------------------------------------------------------

def convert_to_tf_dataset(dataset_map):

    key_map = {'inputs': dataset_map['scaled_inputs'],
               'outputs': dataset_map['scaled_outputs'],
               'active_entries': dataset_map['active_entries'],
               'sequence_lengths': dataset_map['sequence_lengths']}

    if 'propensity_weights' in dataset_map:
        key_map['propensity_weights'] = dataset_map['propensity_weights']

    if 'initial_states' in dataset_map:
        key_map['initial_states'] = dataset_map['initial_states']

    tf_dataset = tf.data.Dataset.from_tensor_slices(key_map)

    return tf_dataset


def get_processed_data(raw_sim_data,
                       scaling_params,
                       b_predict_actions,
                       b_use_actions_only,
                       b_predict_censoring):
    """
    Create formatted data to train both propensity networks and seq2seq architecture

    :param raw_sim_data: Data from simulation
    :param scaling_params: means/standard deviations to normalise the data to
    :param b_predict_actions: flag to package data for propensity network to forecast actions
    :param b_use_actions_only:  flag to package data with only action inputs and not covariates
    :param b_predict_censoring: flag to package data to predict censoring locations
    :return: processed data to train specific network
    """

    # checks
    if b_predict_actions and b_predict_censoring:
        raise ValueError("problem with RNN! RNN is both actions and censoring")

    mean, std = scaling_params

    horizon = 1
    offset = 1

    mean['chemo_application'] = 0
    mean['radio_application'] = 0
    std['chemo_application'] = 1
    std['radio_application'] = 1

    # Continuous values
    cancer_volume = (raw_sim_data['cancer_volume'] - mean['cancer_volume']) / std['cancer_volume']
    patient_types = (raw_sim_data['patient_types'] - mean['patient_types']) / std['patient_types']

    patient_types = np.stack([patient_types for t in range(cancer_volume.shape[1])], axis=1)

    # Binary application
    chemo_application = raw_sim_data['chemo_application']
    radio_application = raw_sim_data['radio_application']
    death_flags = raw_sim_data['death_flags']
    recovery_flags = raw_sim_data['recovery_flags']
    active_flags = (death_flags + recovery_flags == 0.0) * 1
    sequence_lengths = raw_sim_data['sequence_lengths']

    # Parcelling INPUTS
    if b_predict_actions:
        if b_use_actions_only:
            inputs = np.concatenate([chemo_application[:, :, np.newaxis],
                                     radio_application[:, :, np.newaxis]],
                                    axis=2)
            inputs = inputs[:, :-offset, :]

            actions = inputs.copy()
            input_means = 0
            input_stds = 1

        else:
            # Uses current covariate, to remove confounding effects between action and current value
            inputs = np.concatenate([cancer_volume[:, 1:, np.newaxis],  # conditioned on value
                                     patient_types[:, :-1, np.newaxis],
                                     chemo_application[:, :-1, np.newaxis],
                                     radio_application[:, :-1, np.newaxis]],
                                    axis=2)

            actions = inputs[:, :, -2:].copy()

            input_means = mean[
                ['cancer_volume', 'patient_types', 'chemo_application', 'radio_application']].values.flatten()
            input_stds = std[
                ['cancer_volume', 'patient_types', 'chemo_application', 'radio_application']].values.flatten()

    elif b_predict_censoring:
        if b_use_actions_only:
            inputs = np.concatenate([chemo_application[:, :, np.newaxis],
                                     radio_application[:, :, np.newaxis]],
                                    axis=2)
            inputs = inputs[:, :-offset, :]

            actions = inputs.copy()

            input_means = 0
            input_stds = 1

        else:
            # Censoring only uses past history
            inputs = np.concatenate([cancer_volume[:, :, np.newaxis],  # conditioned on value
                                     patient_types[:, :, np.newaxis],
                                     chemo_application[:, :, np.newaxis],
                                     radio_application[:, :, np.newaxis]],
                                    axis=2)
            inputs = inputs[:, :-offset, :]

            actions = inputs[:, :, -2:].copy()

            input_means = mean[
                ['cancer_volume', 'patient_types', 'chemo_application', 'radio_application']].values.flatten()
            input_stds = std[
                ['cancer_volume', 'patient_types', 'chemo_application', 'radio_application']].values.flatten()
    else:
        inputs = np.concatenate([cancer_volume[:, :, np.newaxis],  # conditioned on value
                                 patient_types[:, :, np.newaxis],
                                 chemo_application[:, :, np.newaxis],
                                 radio_application[:, :, np.newaxis]],
                                axis=2)
        inputs = inputs[:, :-offset, :]

        actions = inputs[:, :, -2:].copy()

        input_means = mean[
            ['cancer_volume', 'patient_types', 'chemo_application', 'radio_application']].values.flatten()
        input_stds = std[['cancer_volume', 'patient_types', 'chemo_application', 'radio_application']].values.flatten()

    # Parcelling OUTPUTS
    if b_predict_actions:
        outputs = np.concatenate([chemo_application[:, :, np.newaxis],
                                  radio_application[:, :, np.newaxis]],
                                 axis=2)

        outputs = outputs[:, 1:, :]
        output_means = 0
        output_stds = 1

    elif b_predict_censoring:

        outputs = np.concatenate([active_flags[:, :, np.newaxis]],
                                 axis=2)
        output_means = 0
        output_stds = 1
        outputs = outputs[:, 1:, :]
    else:

        (patient_num, num_time_steps) = cancer_volume.shape
        outputs = np.zeros((patient_num, num_time_steps - 1, horizon))

        for h in range(horizon):
            outputs[:, :num_time_steps - 1 - h, h] = cancer_volume[:, h + 1:]

        output_means = mean[['cancer_volume']].values.flatten()[0]  # because we only need scalars here
        output_stds = std[['cancer_volume']].values.flatten()[0]

    # Set array alignment
    sequence_lengths = np.array([i - 1 for i in sequence_lengths]) # everything shortens by 1

    # Remove any trajectories that are too short
    inputs = inputs[sequence_lengths > 0, :, :]
    outputs = outputs[sequence_lengths > 0, :, :]
    sequence_lengths = sequence_lengths[sequence_lengths > 0]
    actions = actions[sequence_lengths > 0, :, :]

    # Add active entires
    active_entries = np.zeros(outputs.shape)

    for i in range(sequence_lengths.shape[0]):
        sequence_length = int(sequence_lengths[i])

        if not b_predict_actions:
            for k in range(horizon):
                #include the censoring point too, but ignore future shifts that don't exist
                active_entries[i, :sequence_length-k, k] = 1
        else:
            active_entries[i, :sequence_length, :] = 1

    return {'outputs': (outputs * std['cancer_volume'] + mean['cancer_volume'])
                        if not (b_predict_actions or b_predict_censoring) else outputs,  # already scaled
            'scaled_inputs': inputs,
            'scaled_outputs': outputs,
            'actions': actions,
            'sequence_lengths': sequence_lengths,
            'active_entries': active_entries,
            'input_means': input_means,
            'inputs_stds': input_stds,
            'output_means': output_means,
            'output_stds': output_stds
            }


def get_cancer_sim_data(chemo_coeff, radio_coeff, b_load,  b_save=False, seed=100, model_root=MODEL_ROOT, window_size=15):

    if window_size == 15:  # default 3 week (business days) window used
        pickle_file = os.path.join(model_root, 'cancer_sim_{}_{}.p'.format(chemo_coeff, radio_coeff))
    else:
        pickle_file = os.path.join(model_root, 'cancer_sim_{}_{}_{}.p'.format(chemo_coeff, radio_coeff, window_size))

    def _generate():
        num_time_steps = 60  # about half a year
        np.random.seed(seed)
        num_patients = 10000

        params = sim.get_confounding_params(num_patients, chemo_coeff=chemo_coeff,
                                            radio_coeff=radio_coeff)
        params['window_size'] = window_size
        training_data = sim.simulate(params, num_time_steps)

        params = sim.get_confounding_params(int(num_patients / 10), chemo_coeff=chemo_coeff,
                                            radio_coeff=radio_coeff)
        params['window_size'] = window_size
        validation_data = sim.simulate(params, num_time_steps)

        params = sim.get_confounding_params(int(num_patients / 10), chemo_coeff=chemo_coeff,
                                            radio_coeff=radio_coeff)
        params['window_size'] = window_size
        test_data = sim.simulate(params, num_time_steps)

        scaling_data = sim.get_scaling_params(training_data)

        pickle_map = {'chemo_coeff': chemo_coeff,
                      'radio_coeff': radio_coeff,
                      'num_time_steps': num_time_steps,
                      'training_data': training_data,
                      'validation_data': validation_data,
                      'test_data': test_data,
                      'scaling_data': scaling_data,
                      'window_size': window_size}

        logging.info("Saving pickle map to {}".format(pickle_file))
        if b_save:
            pickle.dump(pickle_map, open(pickle_file, 'wb'))
        return pickle_map

    # Controls whether to regenerate the data, or load from a persisted file
    if not b_load:

        pickle_map = _generate()

    else:
        logging.info("Loading pickle map from {}".format(pickle_file))

        try:
            pickle_map = pickle.load(open(pickle_file, "rb"))

        except IOError:
            logging.info("Pickle file does not exist, regenerating: {}".format(pickle_file))
            _generate()
            pickle_map = _generate()

    return pickle_map

