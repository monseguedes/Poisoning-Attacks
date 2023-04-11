\ Model Poisoning_Attack
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 2.09193167615609 weights_num[1] - 2.442915335832986 weights_num[2]
   - 2.979030690181919 bias + 0 zn_bias_times_numsample[1,1]
   + 0 zn_bias_times_numsample[1,2] + 0.4863334165213865 Constant + [
   4.814814814814815 weights_num[1] ^2
   + 11.39259259259259 weights_num[1] * weights_num[2]
   + 13.77777777777778 weights_num[1] * bias
   + 6.868333333333334 weights_num[2] ^2
   + 16.33333333333334 weights_num[2] * bias + 10 bias ^2 ] / 2 
Subject To
 cons_first_order_optimality_conditions_num_weights[1]:
   6.137263794406651 weights_num[1] + 5.696296296296296 weights_num[2]
   + 8.444444444444445 bias + 2 tnn_ln_times_numsamples[1,1,1]
   + 2 tnn_ln_times_numsamples[1,2,1] = 3.647487231711644
 cons_first_order_optimality_conditions_num_weights[2]:
   5.696296296296295 weights_num[1] + 8.190782312925169 weights_num[2]
   + 9.766666666666667 bias + 2 tnn_ln_times_numsamples[1,1,2]
   + 2 tnn_ln_times_numsamples[1,2,2] = 4.042915335832987
 cons_first_order_optimality_conditions_bias:
   1.148148148148148 weights_num[1] + 1.361111111111111 weights_num[2]
   + 2 bias + 0.3333333333333333 ln_numweight_times_numsample[1,1]
   + 0.3333333333333333 ln_numweight_times_numsample[1,2]
   = 0.8298384483636532
 cons_tnn[1,1,1]: - tnn_ln_times_numsamples[1,1,1]
   + 0.7777777777777778 ln_numweight_times_numsample[1,1] = 0
 cons_tnn[1,1,2]: - tnn_ln_times_numsamples[1,1,2]
   + 0.8 ln_numweight_times_numsample[1,1] = 0
 cons_tnn[1,2,1]: - tnn_ln_times_numsamples[1,2,1]
   + 0.7777777777777778 ln_numweight_times_numsample[1,2] = 0
 cons_tnn[1,2,2]: - tnn_ln_times_numsamples[1,2,2]
   + 0.8 ln_numweight_times_numsample[1,2] = 0
 cons_ln: 0.7777777777777778 weights_num[1]
   - ln_numweight_times_numsample[1,1] = 0
 cons_ln: 0.8 weights_num[2] - ln_numweight_times_numsample[1,2] = 0
Bounds
 -0.6320275157615231 <= weights_num[1] <= 0.6320275157615231
 -0.6320275157615231 <= weights_num[2] <= 0.6320275157615231
 -0.6320275157615231 <= bias <= 0.6320275157615231
 -0.6320275157615231 <= tnn_ln_times_numsamples[1,1,1] <= 0.6320275157615231
 -0.6320275157615231 <= tnn_ln_times_numsamples[1,1,2] <= 0.6320275157615231
 -0.6320275157615231 <= tnn_ln_times_numsamples[1,2,1] <= 0.6320275157615231
 -0.6320275157615231 <= tnn_ln_times_numsamples[1,2,2] <= 0.6320275157615231
 -0.6320275157615231 <= ln_numweight_times_numsample[1,1] <= 0.6320275157615231
 -0.6320275157615231 <= ln_numweight_times_numsample[1,2] <= 0.6320275157615231
 -0.6320275157615231 <= zn_bias_times_numsample[1,1] <= 0.6320275157615231
 -0.6320275157615231 <= zn_bias_times_numsample[1,2] <= 0.6320275157615231
 Constant = 1
End
