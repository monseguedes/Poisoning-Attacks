\ Model Poisoning_Attack
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 0.0915393181724481 weights_num[1] - 0.0286195402585681 weights_num[2]
   - 0.3499529386350663 weights_num[3] - 0.2971410220802667 weights_num[4]
   - 0.4006360831527023 weights_num[5]
   - 0.1934036939313983 self.weights_cat[1,1]
   - 0.0543257880849882 self.weights_cat[1,2] - 0.5152895431190113 bias
   + 0.0756952124468664 Constant + [ 0.0667102139611212 weights_num[1] ^2
   + 0.0405245484005005 weights_num[1] * weights_num[2]
   + 0.4315524733954583 weights_num[1] * weights_num[3]
   + 0.3922516766849545 weights_num[1] * weights_num[4]
   + 0.5136921295320541 weights_num[1] * weights_num[5]
   + 0.2233218892931689 weights_num[1] * self.weights_cat[1,1]
   + 0.0623287671232876 weights_num[1] * self.weights_cat[1,2]
   + 0.6795204909151052 weights_num[1] * bias
   + 0.00894396035398838 weights_num[2] ^2
   + 0.131982207265107 weights_num[2] * weights_num[3]
   + 0.1319639393302015 weights_num[2] * weights_num[4]
   + 0.1457089777577265 weights_num[2] * weights_num[5]
   + 0.0577241814485031 weights_num[2] * self.weights_cat[1,1]
   + 0.0558741732688307 weights_num[2] * self.weights_cat[1,2]
   + 0.2133352029727264 weights_num[2] * bias
   + 0.8469135802469143 weights_num[3] ^2
   + 1.447222222222223 weights_num[3] * weights_num[4]
   + 2.022705314009663 weights_num[3] * weights_num[5]
   + 0.9333333333333338 weights_num[3] * self.weights_cat[1,1]
   + 0.2444444444444446 weights_num[3] * self.weights_cat[1,2]
   + 2.533333333333335 weights_num[3] * bias + 0.709375 weights_num[4] ^2
   + 1.719746376811594 weights_num[4] * weights_num[5]
   + 0.825 weights_num[4] * self.weights_cat[1,1]
   + 0.275 weights_num[4] * self.weights_cat[1,2]
   + 2.3 weights_num[4] * bias + 1.293557025834909 weights_num[5] ^2
   + 1.224637681159421 weights_num[5] * self.weights_cat[1,1]
   + 0.2347826086956522 weights_num[5] * self.weights_cat[1,2]
   + 3.08840579710145 weights_num[5] * bias + 0.7 self.weights_cat[1,1] ^2
   + 1.4 self.weights_cat[1,1] * bias + 0.2 self.weights_cat[1,2] ^2
   + 0.4 self.weights_cat[1,2] * bias + 2 bias ^2 ] / 2 
Subject To
 SOSconstraints[1,_1]: x_poison_cat[1,1,1] + x_poison_cat[1,1,2]
   + x_poison_cat[1,1,3] = 1
 cons_first_order_optimality_conditions_num_weights[1]:
   0.0835335371058297 weights_num[1] + 0.0192974040002383 weights_num[2]
   + 0.2055011778073611 weights_num[3] + 0.1867865127071212 weights_num[4]
   + 0.2446152997771686 weights_num[5]
   + 0.1063437568062709 self.weights_cat[1,1]
   + 0.0296803652968036 self.weights_cat[1,2] + 0.32358118615005 bias
   + 0.0952380952380952 tnn_ln_times_numsamples[1,1,1]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,2,1]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,3,1]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,4,1]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,5,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,1,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,2,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,3,1]
   + 0.0952380952380952 zn_bias_times_numsample[1,1] = 0.0871803030213791
 cons_first_order_optimality_conditions_num_weights[2]:
   0.0192974040002383 weights_num[1] + 0.0285180574799889 weights_num[2]
   + 0.0628486701262414 weights_num[3] + 0.0628399711096198 weights_num[4]
   + 0.0693852275036793 weights_num[5]
   + 0.0274877054516681 self.weights_cat[1,1]
   + 0.0266067491756337 self.weights_cat[1,2] + 0.1015881918917745 bias
   + 0.0952380952380952 tnn_ln_times_numsamples[1,1,2]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,2,2]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,3,2]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,4,2]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,5,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,1,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,2,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,3,2]
   + 0.0952380952380952 zn_bias_times_numsample[1,2] = 0.0274258628580303
 cons_first_order_optimality_conditions_num_weights[4]:
   0.1867865127071211 weights_num[1] + 0.0628399711096198 weights_num[2]
   + 0.6891534391534393 weights_num[3] + 0.695595238095238 weights_num[4]
   + 0.8189268461007592 weights_num[5]
   + 0.3928571428571428 self.weights_cat[1,1]
   + 0.1309523809523809 self.weights_cat[1,2] + 1.095238095238096 bias
   + 0.0952380952380952 tnn_ln_times_numsamples[1,1,4]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,2,4]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,3,4]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,4,4]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,5,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,1,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,2,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,3,4]
   + 0.0952380952380952 zn_bias_times_numsample[1,4] = 0.3663247829335873
 cons_first_order_optimality_conditions_num_weights[5]:
   0.2446152997771686 weights_num[1] + 0.0693852275036793 weights_num[2]
   + 0.9631930066712679 weights_num[3] + 0.8189268461007591 weights_num[4]
   + 1.251959072223723 weights_num[5]
   + 0.5831608005521051 self.weights_cat[1,1]
   + 0.1118012422360249 self.weights_cat[1,2] + 1.470669427191166 bias
   + 0.0952380952380952 tnn_ln_times_numsamples[1,1,5]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,2,5]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,3,5]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,4,5]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,5,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,1,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,2,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,3,5]
   + 0.0952380952380952 zn_bias_times_numsample[1,5] = 0.4505712869225184
 cons_first_order_optimality_conditions_pnum_weights[3]:
   - 0.0952380952380952 x_poison_num[1,3]
   + 0.2055011778073611 weights_num[1] + 0.0628486701262414 weights_num[2]
   + 0.8265843621399183 weights_num[3] + 0.6891534391534394 weights_num[4]
   + 0.9631930066712679 weights_num[5]
   + 0.4444444444444447 self.weights_cat[1,1]
   + 0.1164021164021165 self.weights_cat[1,2] + 1.206349206349207 bias
   + 0.0952380952380952 tnn_ln_times_numsamples[1,1,3]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,2,3]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,3,3]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,4,3]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,5,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,1,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,2,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,3,3]
   + 0.0952380952380952 zn_bias_times_numsample[1,3] = 0.3332885129857774
 cons_first_order_optimality_conditions_pcat_weights[1,_1]:
   - 0.0952380952380952 x_poison_cat[1,1,1]
   + 0.1063437568062709 weights_num[1] + 0.0274877054516681 weights_num[2]
   + 0.4444444444444446 weights_num[3] + 0.3928571428571428 weights_num[4]
   + 0.583160800552105 weights_num[5]
   + 0.6866666666666666 self.weights_cat[1,1] + 0.6666666666666666 bias
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,1,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,1,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,1,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,1,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,1,1]
   + 0.0952380952380952 zc_bias_times_catsample[1,1,1]
   = 0.1841939942203793
 cons_first_order_optimality_conditions_pcat_weights[1,_2]:
   - 0.0952380952380952 x_poison_cat[1,1,2]
   + 0.0296803652968036 weights_num[1] + 0.0266067491756337 weights_num[2]
   + 0.1164021164021165 weights_num[3] + 0.1309523809523809 weights_num[4]
   + 0.1118012422360249 weights_num[5]
   + 0.2104761904761905 self.weights_cat[1,2] + 0.1904761904761905 bias
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,1,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,1,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,1,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,1,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,1,2]
   + 0.0952380952380952 zc_bias_times_catsample[1,1,2]
   = 0.0517388457952269
 cons_first_order_optimality_conditions_pcat_weights[1,_3]:
   - 0.0952380952380952 x_poison_cat[1,1,3] + 0.02 self.weights_cat[1,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,1,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,1,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,1,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,1,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,1,3]
   + 0.0952380952380952 zc_bias_times_catsample[1,1,3] = 0
 cons_first_order_optimality_conditions_bias:
   0.32358118615005 weights_num[1] + 0.1015881918917745 weights_num[2]
   + 1.206349206349207 weights_num[3] + 1.095238095238095 weights_num[4]
   + 1.470669427191166 weights_num[5]
   + 0.6666666666666666 self.weights_cat[1,1]
   + 0.1904761904761905 self.weights_cat[1,2] + 2 bias
   + 0.0952380952380952 ln_numweight_times_numsample[1,1]
   + 0.0952380952380952 ln_numweight_times_numsample[1,2]
   + 0.0952380952380952 ln_numweight_times_numsample[1,3]
   + 0.0952380952380952 ln_numweight_times_numsample[1,4]
   + 0.0952380952380952 ln_numweight_times_numsample[1,5]
   + 0.0952380952380952 lc_catweight_times_catsample[1,1,1]
   + 0.0952380952380952 lc_catweight_times_catsample[1,1,2]
   + 0.0952380952380952 lc_catweight_times_catsample[1,1,3]
   = 0.5859900410657251
 cons_tnn: - tnn_ln_times_numsamples[1,1,1] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,1,2]
   + 0.0017761574236369 ln_numweight_times_numsample[1,1] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,1,4]
   + 0.875 ln_numweight_times_numsample[1,1] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,1,5]
   + 0.72463768115942 ln_numweight_times_numsample[1,1] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,2,1] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,2,2]
   + 0.0017761574236369 ln_numweight_times_numsample[1,2] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,2,4]
   + 0.875 ln_numweight_times_numsample[1,2] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,2,5]
   + 0.72463768115942 ln_numweight_times_numsample[1,2] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,3,1] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,3,2]
   + 0.0017761574236369 ln_numweight_times_numsample[1,3] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,3,4]
   + 0.875 ln_numweight_times_numsample[1,3] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,3,5]
   + 0.72463768115942 ln_numweight_times_numsample[1,3] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,4,1] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,4,2]
   + 0.0017761574236369 ln_numweight_times_numsample[1,4] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,4,4]
   + 0.875 ln_numweight_times_numsample[1,4] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,4,5]
   + 0.72463768115942 ln_numweight_times_numsample[1,4] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,5,1] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,5,2]
   + 0.0017761574236369 ln_numweight_times_numsample[1,5] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,5,4]
   + 0.875 ln_numweight_times_numsample[1,5] = 0
 cons_tnn: - tnn_ln_times_numsamples[1,5,5]
   + 0.72463768115942 ln_numweight_times_numsample[1,5] = 0
 cons_tcn: - tcn_lc_times_numsamples[1,1,1,1] = 0
 cons_tcn: - tcn_lc_times_numsamples[1,1,1,2]
   + 0.0017761574236369 lc_catweight_times_catsample[1,1,1] = 0
 cons_tcn: - tcn_lc_times_numsamples[1,1,1,4]
   + 0.875 lc_catweight_times_catsample[1,1,1] = 0
 cons_tcn: - tcn_lc_times_numsamples[1,1,1,5]
   + 0.72463768115942 lc_catweight_times_catsample[1,1,1] = 0
 cons_tcn: - tcn_lc_times_numsamples[1,1,2,1] = 0
 cons_tcn: - tcn_lc_times_numsamples[1,1,2,2]
   + 0.0017761574236369 lc_catweight_times_catsample[1,1,2] = 0
 cons_tcn: - tcn_lc_times_numsamples[1,1,2,4]
   + 0.875 lc_catweight_times_catsample[1,1,2] = 0
 cons_tcn: - tcn_lc_times_numsamples[1,1,2,5]
   + 0.72463768115942 lc_catweight_times_catsample[1,1,2] = 0
 cons_tcn: - tcn_lc_times_numsamples[1,1,3,1] = 0
 cons_tcn: - tcn_lc_times_numsamples[1,1,3,2]
   + 0.0017761574236369 lc_catweight_times_catsample[1,1,3] = 0
 cons_tcn: - tcn_lc_times_numsamples[1,1,3,4]
   + 0.875 lc_catweight_times_catsample[1,1,3] = 0
 cons_tcn: - tcn_lc_times_numsamples[1,1,3,5]
   + 0.72463768115942 lc_catweight_times_catsample[1,1,3] = 0
 cons_ln: - ln_numweight_times_numsample[1,1] = 0
 cons_ln: 0.0017761574236369 weights_num[2]
   - ln_numweight_times_numsample[1,2] = 0
 cons_ln: 0.875 weights_num[4] - ln_numweight_times_numsample[1,4] = 0
 cons_ln: 0.72463768115942 weights_num[5]
   - ln_numweight_times_numsample[1,5] = 0
 cons_zn: - zn_bias_times_numsample[1,1] = 0
 cons_zn: 0.0017761574236369 bias - zn_bias_times_numsample[1,2] = 0
 cons_zn: 0.875 bias - zn_bias_times_numsample[1,4] = 0
 cons_zn: 0.72463768115942 bias - zn_bias_times_numsample[1,5] = 0
 pcons_tnn: - tnn_ln_times_numsamples[1,1,3] + [
   x_poison_num[1,3] * ln_numweight_times_numsample[1,1] ] = 0
 pcons_tnn: - tnn_ln_times_numsamples[1,2,3] + [
   x_poison_num[1,3] * ln_numweight_times_numsample[1,2] ] = 0
 pcons_tnn: - tnn_ln_times_numsamples[1,3,3] + [
   x_poison_num[1,3] * ln_numweight_times_numsample[1,3] ] = 0
 pcons_tnn: - tnn_ln_times_numsamples[1,4,3] + [
   x_poison_num[1,3] * ln_numweight_times_numsample[1,4] ] = 0
 pcons_tnn: - tnn_ln_times_numsamples[1,5,3] + [
   x_poison_num[1,3] * ln_numweight_times_numsample[1,5] ] = 0
 pcons_tcn: - tcn_lc_times_numsamples[1,1,1,3] + [
   x_poison_num[1,3] * lc_catweight_times_catsample[1,1,1] ] = 0
 pcons_tcn: - tcn_lc_times_numsamples[1,1,2,3] + [
   x_poison_num[1,3] * lc_catweight_times_catsample[1,1,2] ] = 0
 pcons_tcn: - tcn_lc_times_numsamples[1,1,3,3] + [
   x_poison_num[1,3] * lc_catweight_times_catsample[1,1,3] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,1,1,1] + [
   x_poison_cat[1,1,1] * ln_numweight_times_numsample[1,1] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,1,1,2] + [
   x_poison_cat[1,1,2] * ln_numweight_times_numsample[1,1] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,1,1,3] + [
   x_poison_cat[1,1,3] * ln_numweight_times_numsample[1,1] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,2,1,1] + [
   x_poison_cat[1,1,1] * ln_numweight_times_numsample[1,2] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,2,1,2] + [
   x_poison_cat[1,1,2] * ln_numweight_times_numsample[1,2] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,2,1,3] + [
   x_poison_cat[1,1,3] * ln_numweight_times_numsample[1,2] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,3,1,1] + [
   x_poison_cat[1,1,1] * ln_numweight_times_numsample[1,3] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,3,1,2] + [
   x_poison_cat[1,1,2] * ln_numweight_times_numsample[1,3] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,3,1,3] + [
   x_poison_cat[1,1,3] * ln_numweight_times_numsample[1,3] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,4,1,1] + [
   x_poison_cat[1,1,1] * ln_numweight_times_numsample[1,4] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,4,1,2] + [
   x_poison_cat[1,1,2] * ln_numweight_times_numsample[1,4] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,4,1,3] + [
   x_poison_cat[1,1,3] * ln_numweight_times_numsample[1,4] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,5,1,1] + [
   x_poison_cat[1,1,1] * ln_numweight_times_numsample[1,5] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,5,1,2] + [
   x_poison_cat[1,1,2] * ln_numweight_times_numsample[1,5] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,5,1,3] + [
   x_poison_cat[1,1,3] * ln_numweight_times_numsample[1,5] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,1,1,1] + [
   x_poison_cat[1,1,1] * lc_catweight_times_catsample[1,1,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,1,1,2] + [
   x_poison_cat[1,1,2] * lc_catweight_times_catsample[1,1,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,1,1,3] + [
   x_poison_cat[1,1,3] * lc_catweight_times_catsample[1,1,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,2,1,1] + [
   x_poison_cat[1,1,1] * lc_catweight_times_catsample[1,1,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,2,1,2] + [
   x_poison_cat[1,1,2] * lc_catweight_times_catsample[1,1,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,2,1,3] + [
   x_poison_cat[1,1,3] * lc_catweight_times_catsample[1,1,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,3,1,1] + [
   x_poison_cat[1,1,1] * lc_catweight_times_catsample[1,1,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,3,1,2] + [
   x_poison_cat[1,1,2] * lc_catweight_times_catsample[1,1,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,3,1,3] + [
   x_poison_cat[1,1,3] * lc_catweight_times_catsample[1,1,3] ] = 0
 pcons_ln: - ln_numweight_times_numsample[1,3] + [
   x_poison_num[1,3] * weights_num[3] ] = 0
 pcons_lc: - lc_catweight_times_catsample[1,1,1] + [
   x_poison_cat[1,1,1] * self.weights_cat[1,1] ] = 0
 pcons_lc: - lc_catweight_times_catsample[1,1,2] + [
   x_poison_cat[1,1,2] * self.weights_cat[1,2] ] = 0
 pcons_lc: - lc_catweight_times_catsample[1,1,3] + [
   x_poison_cat[1,1,3] * self.weights_cat[1,3] ] = 0
 pcons_zn: - zn_bias_times_numsample[1,3] + [ x_poison_num[1,3] * bias ]
   = 0
 pcons_zc: - zc_bias_times_catsample[1,1,1] + [ x_poison_cat[1,1,1] * bias
   ] = 0
 pcons_zc: - zc_bias_times_catsample[1,1,2] + [ x_poison_cat[1,1,2] * bias
   ] = 0
 pcons_zc: - zc_bias_times_catsample[1,1,3] + [ x_poison_cat[1,1,3] * bias
   ] = 0
Bounds
 x_poison_num[1,3] <= 1
 -0.3987484840607427 <= weights_num[1] <= 0.3987484840607427
 -0.3987484840607427 <= weights_num[2] <= 0.3987484840607427
 -0.3987484840607427 <= weights_num[3] <= 0.3987484840607427
 -0.3987484840607427 <= weights_num[4] <= 0.3987484840607427
 -0.3987484840607427 <= weights_num[5] <= 0.3987484840607427
 -0.3987484840607427 <= self.weights_cat[1,1] <= 0.3987484840607427
 -0.3987484840607427 <= self.weights_cat[1,2] <= 0.3987484840607427
 -0.3987484840607427 <= self.weights_cat[1,3] <= 0.3987484840607427
 -0.3987484840607427 <= bias <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,1,1] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,1,2] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,1,3] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,1,4] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,1,5] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,2,1] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,2,2] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,2,3] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,2,4] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,2,5] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,3,1] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,3,2] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,3,3] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,3,4] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,3,5] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,4,1] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,4,2] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,4,3] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,4,4] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,4,5] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,5,1] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,5,2] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,5,3] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,5,4] <= 0.3987484840607427
 -0.3987484840607427 <= tnn_ln_times_numsamples[1,5,5] <= 0.3987484840607427
 -0.3987484840607427 <= tcn_lc_times_numsamples[1,1,1,1] <= 0.3987484840607427
 -0.3987484840607427 <= tcn_lc_times_numsamples[1,1,2,1] <= 0.3987484840607427
 -0.3987484840607427 <= tcn_lc_times_numsamples[1,1,3,1] <= 0.3987484840607427
 -0.3987484840607427 <= tcn_lc_times_numsamples[1,1,1,2] <= 0.3987484840607427
 -0.3987484840607427 <= tcn_lc_times_numsamples[1,1,2,2] <= 0.3987484840607427
 -0.3987484840607427 <= tcn_lc_times_numsamples[1,1,3,2] <= 0.3987484840607427
 -0.3987484840607427 <= tcn_lc_times_numsamples[1,1,1,3] <= 0.3987484840607427
 -0.3987484840607427 <= tcn_lc_times_numsamples[1,1,2,3] <= 0.3987484840607427
 -0.3987484840607427 <= tcn_lc_times_numsamples[1,1,3,3] <= 0.3987484840607427
 -0.3987484840607427 <= tcn_lc_times_numsamples[1,1,1,4] <= 0.3987484840607427
 -0.3987484840607427 <= tcn_lc_times_numsamples[1,1,2,4] <= 0.3987484840607427
 -0.3987484840607427 <= tcn_lc_times_numsamples[1,1,3,4] <= 0.3987484840607427
 -0.3987484840607427 <= tcn_lc_times_numsamples[1,1,1,5] <= 0.3987484840607427
 -0.3987484840607427 <= tcn_lc_times_numsamples[1,1,2,5] <= 0.3987484840607427
 -0.3987484840607427 <= tcn_lc_times_numsamples[1,1,3,5] <= 0.3987484840607427
 -0.3987484840607427 <= tnc_ln_times_catsamples[1,1,1,1] <= 0.3987484840607427
 -0.3987484840607427 <= tnc_ln_times_catsamples[1,1,1,2] <= 0.3987484840607427
 -0.3987484840607427 <= tnc_ln_times_catsamples[1,1,1,3] <= 0.3987484840607427
 -0.3987484840607427 <= tnc_ln_times_catsamples[1,2,1,1] <= 0.3987484840607427
 -0.3987484840607427 <= tnc_ln_times_catsamples[1,2,1,2] <= 0.3987484840607427
 -0.3987484840607427 <= tnc_ln_times_catsamples[1,2,1,3] <= 0.3987484840607427
 -0.3987484840607427 <= tnc_ln_times_catsamples[1,3,1,1] <= 0.3987484840607427
 -0.3987484840607427 <= tnc_ln_times_catsamples[1,3,1,2] <= 0.3987484840607427
 -0.3987484840607427 <= tnc_ln_times_catsamples[1,3,1,3] <= 0.3987484840607427
 -0.3987484840607427 <= tnc_ln_times_catsamples[1,4,1,1] <= 0.3987484840607427
 -0.3987484840607427 <= tnc_ln_times_catsamples[1,4,1,2] <= 0.3987484840607427
 -0.3987484840607427 <= tnc_ln_times_catsamples[1,4,1,3] <= 0.3987484840607427
 -0.3987484840607427 <= tnc_ln_times_catsamples[1,5,1,1] <= 0.3987484840607427
 -0.3987484840607427 <= tnc_ln_times_catsamples[1,5,1,2] <= 0.3987484840607427
 -0.3987484840607427 <= tnc_ln_times_catsamples[1,5,1,3] <= 0.3987484840607427
 -0.3987484840607427 <= tcc_lc_times_catsample[1,1,1,1,1] <= 0.3987484840607427
 -0.3987484840607427 <= tcc_lc_times_catsample[1,1,1,1,2] <= 0.3987484840607427
 -0.3987484840607427 <= tcc_lc_times_catsample[1,1,1,1,3] <= 0.3987484840607427
 -0.3987484840607427 <= tcc_lc_times_catsample[1,1,2,1,1] <= 0.3987484840607427
 -0.3987484840607427 <= tcc_lc_times_catsample[1,1,2,1,2] <= 0.3987484840607427
 -0.3987484840607427 <= tcc_lc_times_catsample[1,1,2,1,3] <= 0.3987484840607427
 -0.3987484840607427 <= tcc_lc_times_catsample[1,1,3,1,1] <= 0.3987484840607427
 -0.3987484840607427 <= tcc_lc_times_catsample[1,1,3,1,2] <= 0.3987484840607427
 -0.3987484840607427 <= tcc_lc_times_catsample[1,1,3,1,3] <= 0.3987484840607427
 -0.3987484840607427 <= ln_numweight_times_numsample[1,1] <= 0.3987484840607427
 -0.3987484840607427 <= ln_numweight_times_numsample[1,2] <= 0.3987484840607427
 -0.3987484840607427 <= ln_numweight_times_numsample[1,3] <= 0.3987484840607427
 -0.3987484840607427 <= ln_numweight_times_numsample[1,4] <= 0.3987484840607427
 -0.3987484840607427 <= ln_numweight_times_numsample[1,5] <= 0.3987484840607427
 -0.3987484840607427 <= lc_catweight_times_catsample[1,1,1] <= 0.3987484840607427
 -0.3987484840607427 <= lc_catweight_times_catsample[1,1,2] <= 0.3987484840607427
 -0.3987484840607427 <= lc_catweight_times_catsample[1,1,3] <= 0.3987484840607427
 -0.3987484840607427 <= zn_bias_times_numsample[1,1] <= 0.3987484840607427
 -0.3987484840607427 <= zn_bias_times_numsample[1,2] <= 0.3987484840607427
 -0.3987484840607427 <= zn_bias_times_numsample[1,3] <= 0.3987484840607427
 -0.3987484840607427 <= zn_bias_times_numsample[1,4] <= 0.3987484840607427
 -0.3987484840607427 <= zn_bias_times_numsample[1,5] <= 0.3987484840607427
 -0.3987484840607427 <= zc_bias_times_catsample[1,1,1] <= 0.3987484840607427
 -0.3987484840607427 <= zc_bias_times_catsample[1,1,2] <= 0.3987484840607427
 -0.3987484840607427 <= zc_bias_times_catsample[1,1,3] <= 0.3987484840607427
 Constant = 1
Binaries
 x_poison_cat[1,1,1] x_poison_cat[1,1,2] x_poison_cat[1,1,3]
End
