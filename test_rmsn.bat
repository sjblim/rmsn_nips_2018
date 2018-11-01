rem ############################################################################################################################
rem # Script to demonstrate sequence to run for R-MSN training
rem #
rem # Notes:
rem # - Full hyperparameter optimisation is skipped right now, such that models trained with optimal params only
rem # - To run with hyperparameter optimisation, see the "EDIT ME!" sections at the top of each script it the seqeuence
rem ############################################################################################################################

echo Commencing R-MSN Training...
echo 1 - Fitting Propensity Networks
python script_rnn_fit.py propensity_networks

echo 2 - Generating Propensity Scores  # models to used defined at the start of script
python script_propensity_generation.py

echo 3 - Fitting R-MSN Encoder 
python script_rnn_fit.py encoder

echo 4 - Fitting R-MSN Decoder (Seq2Seq)
python script_decoder_fit.py

echo Training Complete.

echo Commencing R-MSN Tests...
echo 1 - Testing one-step-ahead (decoder) predictions
python script_rnn_test.py

echo 2 - Testing multi-step (encoder) predictions
python script_decoder_test.py 

echo Testing Complete.
