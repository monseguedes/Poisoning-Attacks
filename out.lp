\ Model Poisoning_Attack
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 0.3499529386350661 weights_num[1] - 0.4072214507244367 weights_num[2]
   - 0.3309401472017775 weights_num[3] - 0.2246169513493497 weights_num[4]
   - 0.2751648838726736 weights_num[5]
   - 0.0284821552562144 self.weights_cat[1,2]
   - 0.3531315095125676 self.weights_cat[1,4]
   - 0.1193167615608943 self.weights_cat[2,1]
   - 0.2196639355645049 self.weights_cat[2,3]
   - 0.0623941119288988 self.weights_cat[3,3]
   - 0.2880988751562282 self.weights_cat[3,4]
   - 0.0714206360227746 self.weights_cat[4,3]
   - 0.1258297458686294 self.weights_cat[4,4]
   - 0.2462019164004999 self.weights_cat[4,5]
   - 0.189890293014859 self.weights_cat[5,2]
   - 0.250229134842383 self.weights_cat[5,3]
   - 0.0751701152617692 self.weights_cat[5,4] - 0.5152895431190111 bias
   + 0.0756952124468664 Constant + [ 0.8469135802469135 weights_num[1] ^2
   + 2.02 weights_num[1] * weights_num[2]
   + 1.562962962962962 weights_num[1] * weights_num[3]
   + 1.051851851851852 weights_num[1] * weights_num[4]
   + 1.328158295281583 weights_num[1] * weights_num[5]
   + 0.1111111111111111 weights_num[1] * self.weights_cat[1,2]
   + 1.733333333333333 weights_num[1] * self.weights_cat[1,4]
   + 0.7111111111111111 weights_num[1] * self.weights_cat[2,1]
   + 1.022222222222222 weights_num[1] * self.weights_cat[2,3]
   + 0.3555555555555556 weights_num[1] * self.weights_cat[3,3]
   + 1.488888888888888 weights_num[1] * self.weights_cat[3,4]
   + 0.2888888888888889 weights_num[1] * self.weights_cat[4,3]
   + 0.6444444444444445 weights_num[1] * self.weights_cat[4,4]
   + 1.244444444444444 weights_num[1] * self.weights_cat[4,5]
   + 0.9111111111111111 weights_num[1] * self.weights_cat[5,2]
   + 1.244444444444444 weights_num[1] * self.weights_cat[5,3]
   + 0.3777777777777778 weights_num[1] * self.weights_cat[5,4]
   + 2.533333333333334 weights_num[1] * bias
   + 1.322416666666667 weights_num[2] ^2
   + 1.855555555555556 weights_num[2] * weights_num[3]
   + 1.250833333333333 weights_num[2] * weights_num[4]
   + 1.571632420091324 weights_num[2] * weights_num[5]
   + 0.1566666666666667 weights_num[2] * self.weights_cat[1,2]
   + 2.096666666666667 weights_num[2] * self.weights_cat[1,4]
   + 1.04 weights_num[2] * self.weights_cat[2,1]
   + 1.246666666666667 weights_num[2] * self.weights_cat[2,3]
   + 0.5166666666666667 weights_num[2] * self.weights_cat[3,3]
   + 1.96 weights_num[2] * self.weights_cat[3,4]
   + 0.17 weights_num[2] * self.weights_cat[4,3]
   + 0.7933333333333334 weights_num[2] * self.weights_cat[4,4]
   + 1.6 weights_num[2] * self.weights_cat[4,5]
   + 1.036666666666667 weights_num[2] * self.weights_cat[5,2]
   + 1.493333333333333 weights_num[2] * self.weights_cat[5,3]
   + 0.5466666666666667 weights_num[2] * self.weights_cat[5,4]
   + 3.076666666666666 weights_num[2] * bias
   + 0.7777777777777777 weights_num[3] ^2
   + 1.038888888888889 weights_num[3] * weights_num[4]
   + 1.262785388127854 weights_num[3] * weights_num[5]
   + 0.2 weights_num[3] * self.weights_cat[1,2]
   + 1.533333333333333 weights_num[3] * self.weights_cat[1,4]
   + 0.6666666666666666 weights_num[3] * self.weights_cat[2,1]
   + 0.9333333333333332 weights_num[3] * self.weights_cat[2,3]
   + 0.3333333333333333 weights_num[3] * self.weights_cat[3,3]
   + 1.266666666666667 weights_num[3] * self.weights_cat[3,4]
   + 0.2666666666666667 weights_num[3] * self.weights_cat[4,3]
   + 0.7333333333333333 weights_num[3] * self.weights_cat[4,4]
   + 1.066666666666667 weights_num[3] * self.weights_cat[4,5]
   + 0.8666666666666666 weights_num[3] * self.weights_cat[5,2]
   + 1.133333333333333 weights_num[3] * self.weights_cat[5,3]
   + 0.4 weights_num[3] * self.weights_cat[5,4] + 2.4 weights_num[3] * bias
   + 0.3638888888888889 weights_num[4] ^2
   + 0.8627853881278541 weights_num[4] * weights_num[5]
   + 0.1333333333333333 weights_num[4] * self.weights_cat[1,2]
   + 1.05 weights_num[4] * self.weights_cat[1,4]
   + 0.4833333333333333 weights_num[4] * self.weights_cat[2,1]
   + 0.6166666666666667 weights_num[4] * self.weights_cat[2,3]
   + 0.2333333333333333 weights_num[4] * self.weights_cat[3,3]
   + 0.8333333333333331 weights_num[4] * self.weights_cat[3,4]
   + 0.2166666666666667 weights_num[4] * self.weights_cat[4,3]
   + 0.5166666666666667 weights_num[4] * self.weights_cat[4,4]
   + 0.6833333333333332 weights_num[4] * self.weights_cat[4,5]
   + 0.65 weights_num[4] * self.weights_cat[5,2]
   + 0.75 weights_num[4] * self.weights_cat[5,3]
   + 0.2333333333333333 weights_num[4] * self.weights_cat[5,4]
   + 1.633333333333334 weights_num[4] * bias
   + 0.5695170294614376 weights_num[5] ^2
   + 0.1 weights_num[5] * self.weights_cat[1,2]
   + 1.438356164383562 weights_num[5] * self.weights_cat[1,4]
   + 0.6883561643835616 weights_num[5] * self.weights_cat[2,1]
   + 0.8 weights_num[5] * self.weights_cat[2,3]
   + 0.25 weights_num[5] * self.weights_cat[3,3]
   + 1.188356164383562 weights_num[5] * self.weights_cat[3,4]
   + 0.2 weights_num[5] * self.weights_cat[4,3]
   + 0.6383561643835618 weights_num[5] * self.weights_cat[4,4]
   + weights_num[5] * self.weights_cat[4,5]
   + 0.8383561643835618 weights_num[5] * self.weights_cat[5,2]
   + 0.95 weights_num[5] * self.weights_cat[5,3]
   + 0.3 weights_num[5] * self.weights_cat[5,4]
   + 2.088356164383562 weights_num[5] * bias + 0.1 self.weights_cat[1,2] ^2
   + 0.2 self.weights_cat[1,2] * self.weights_cat[2,3]
   + 0.2 self.weights_cat[1,2] * self.weights_cat[4,4]
   + 0.2 self.weights_cat[1,2] * self.weights_cat[5,2]
   + 0.2 self.weights_cat[1,2] * bias + 1.3 self.weights_cat[1,4] ^2
   + self.weights_cat[1,4] * self.weights_cat[2,1]
   + 1.2 self.weights_cat[1,4] * self.weights_cat[2,3]
   + 0.2 self.weights_cat[1,4] * self.weights_cat[3,3]
   + 2 self.weights_cat[1,4] * self.weights_cat[3,4]
   + 0.2 self.weights_cat[1,4] * self.weights_cat[4,3]
   + 0.8 self.weights_cat[1,4] * self.weights_cat[4,4]
   + 1.6 self.weights_cat[1,4] * self.weights_cat[4,5]
   + 0.8 self.weights_cat[1,4] * self.weights_cat[5,2]
   + 1.4 self.weights_cat[1,4] * self.weights_cat[5,3]
   + 0.4 self.weights_cat[1,4] * self.weights_cat[5,4]
   + 2.6 self.weights_cat[1,4] * bias + 0.7 self.weights_cat[2,1] ^2
   + 0.4 self.weights_cat[2,1] * self.weights_cat[3,3]
   + 0.8 self.weights_cat[2,1] * self.weights_cat[3,4]
   + 0.6 self.weights_cat[2,1] * self.weights_cat[4,4]
   + 0.6 self.weights_cat[2,1] * self.weights_cat[4,5]
   + 0.6 self.weights_cat[2,1] * self.weights_cat[5,2]
   + 0.8 self.weights_cat[2,1] * self.weights_cat[5,3]
   + 1.4 self.weights_cat[2,1] * bias + 0.7 self.weights_cat[2,3] ^2
   + self.weights_cat[2,3] * self.weights_cat[3,4]
   + 0.2 self.weights_cat[2,3] * self.weights_cat[4,3]
   + 0.2 self.weights_cat[2,3] * self.weights_cat[4,4]
   + self.weights_cat[2,3] * self.weights_cat[4,5]
   + 0.6 self.weights_cat[2,3] * self.weights_cat[5,2]
   + 0.4 self.weights_cat[2,3] * self.weights_cat[5,3]
   + 0.4 self.weights_cat[2,3] * self.weights_cat[5,4]
   + 1.4 self.weights_cat[2,3] * bias + 0.3 self.weights_cat[3,3] ^2
   + 0.2 self.weights_cat[3,3] * self.weights_cat[4,4]
   + 0.4 self.weights_cat[3,3] * self.weights_cat[5,3]
   + 0.2 self.weights_cat[3,3] * self.weights_cat[5,4]
   + 0.6 self.weights_cat[3,3] * bias + 1.1 self.weights_cat[3,4] ^2
   + 0.4 self.weights_cat[3,4] * self.weights_cat[4,4]
   + 1.8 self.weights_cat[3,4] * self.weights_cat[4,5]
   + 0.6 self.weights_cat[3,4] * self.weights_cat[5,2]
   + 1.2 self.weights_cat[3,4] * self.weights_cat[5,3]
   + 0.4 self.weights_cat[3,4] * self.weights_cat[5,4]
   + 2.2 self.weights_cat[3,4] * bias + 0.2 self.weights_cat[4,3] ^2
   + 0.4 self.weights_cat[4,3] * self.weights_cat[5,2]
   + 0.4 self.weights_cat[4,3] * bias + 0.6 self.weights_cat[4,4] ^2
   + 0.6 self.weights_cat[4,4] * self.weights_cat[5,2]
   + 0.6 self.weights_cat[4,4] * self.weights_cat[5,3]
   + 1.2 self.weights_cat[4,4] * bias + 0.9 self.weights_cat[4,5] ^2
   + 0.4 self.weights_cat[4,5] * self.weights_cat[5,2]
   + self.weights_cat[4,5] * self.weights_cat[5,3]
   + 0.4 self.weights_cat[4,5] * self.weights_cat[5,4]
   + 1.8 self.weights_cat[4,5] * bias + 0.8 self.weights_cat[5,2] ^2
   + 1.6 self.weights_cat[5,2] * bias + 0.9 self.weights_cat[5,3] ^2
   + 1.8 self.weights_cat[5,3] * bias + 0.3 self.weights_cat[5,4] ^2
   + 0.6 self.weights_cat[5,4] * bias + 2 bias ^2 ] / 2 
Subject To
 SOSconstraints[1,2]: x_poison_cat[1,2,1] + x_poison_cat[1,2,2]
   + x_poison_cat[1,2,3] + x_poison_cat[1,2,4] + x_poison_cat[1,2,5] = 1
 cons_first_order_optimality_conditions_num_weights[1]:
   2.129033341731754 weights_num[1] + 0.9619047619047619 weights_num[2]
   + 0.7442680776014109 weights_num[3] + 0.5008818342151676 weights_num[4]
   + 0.6324563310864679 weights_num[5]
   + 0.0529100529100529 self.weights_cat[1,2]
   + 0.8253968253968254 self.weights_cat[1,4]
   + 0.3386243386243386 self.weights_cat[2,1]
   + 0.4867724867724867 self.weights_cat[2,3]
   + 0.1693121693121693 self.weights_cat[3,3]
   + 0.708994708994709 self.weights_cat[3,4]
   + 0.1375661375661376 self.weights_cat[4,3]
   + 0.3068783068783069 self.weights_cat[4,4]
   + 0.5925925925925926 self.weights_cat[4,5]
   + 0.4338624338624338 self.weights_cat[5,2]
   + 0.5925925925925926 self.weights_cat[5,3]
   + 0.1798941798941799 self.weights_cat[5,4] + 1.259259259259259 bias
   + 0.0952380952380952 tnn_ln_times_numsamples[1,1,1]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,2,1]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,3,1]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,4,1]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,5,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,1,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,2,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,3,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,4,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,5,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,1,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,2,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,3,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,4,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,5,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,1,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,2,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,3,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,4,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,1,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,2,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,3,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,4,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,5,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,6,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,1,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,2,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,3,1]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,4,1]
   = 0.339394366890548
 cons_first_order_optimality_conditions_num_weights[2]:
   0.9619047619047618 weights_num[1] + 2.581893424036281 weights_num[2]
   + 0.8835978835978835 weights_num[3] + 0.5956349206349205 weights_num[4]
   + 0.748396390519678 weights_num[5]
   + 0.0746031746031746 self.weights_cat[1,2]
   + 0.9984126984126984 self.weights_cat[1,4]
   + 0.4952380952380953 self.weights_cat[2,1]
   + 0.5936507936507937 self.weights_cat[2,3]
   + 0.246031746031746 self.weights_cat[3,3]
   + 0.9333333333333335 self.weights_cat[3,4]
   + 0.080952380952381 self.weights_cat[4,3]
   + 0.3777777777777778 self.weights_cat[4,4]
   + 0.7619047619047619 self.weights_cat[4,5]
   + 0.4936507936507936 self.weights_cat[5,2]
   + 0.7111111111111111 self.weights_cat[5,3]
   + 0.2603174603174603 self.weights_cat[5,4] + 1.555555555555555 bias
   + 0.0952380952380952 tnn_ln_times_numsamples[1,1,2]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,2,2]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,3,2]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,4,2]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,5,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,1,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,2,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,3,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,4,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,5,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,1,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,2,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,3,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,4,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,5,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,1,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,2,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,3,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,4,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,1,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,2,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,3,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,4,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,5,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,6,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,1,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,2,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,3,2]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,4,2]
   = 0.3982709632480498
 cons_first_order_optimality_conditions_num_weights[3]:
   0.7442680776014109 weights_num[1] + 0.8835978835978836 weights_num[2]
   + 2.063189720332578 weights_num[3] + 0.4947089947089947 weights_num[4]
   + 0.6013263752989779 weights_num[5]
   + 0.0952380952380952 self.weights_cat[1,2]
   + 0.7301587301587301 self.weights_cat[1,4]
   + 0.3174603174603174 self.weights_cat[2,1]
   + 0.4444444444444444 self.weights_cat[2,3]
   + 0.1587301587301587 self.weights_cat[3,3]
   + 0.6031746031746031 self.weights_cat[3,4]
   + 0.126984126984127 self.weights_cat[4,3]
   + 0.3492063492063492 self.weights_cat[4,4]
   + 0.5079365079365079 self.weights_cat[4,5]
   + 0.4126984126984127 self.weights_cat[5,2]
   + 0.5396825396825397 self.weights_cat[5,3]
   + 0.1904761904761905 self.weights_cat[5,4] + 1.174603174603174 bias
   + 0.0952380952380952 tnn_ln_times_numsamples[1,1,3]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,2,3]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,3,3]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,4,3]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,5,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,1,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,2,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,3,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,4,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,5,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,1,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,2,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,3,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,4,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,5,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,1,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,2,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,3,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,4,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,1,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,2,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,3,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,4,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,5,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,6,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,1,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,2,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,3,3]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,4,3]
   = 0.3188446049159839
 cons_first_order_optimality_conditions_num_weights[4]:
   0.5008818342151675 weights_num[1] + 0.5956349206349206 weights_num[2]
   + 0.4947089947089947 weights_num[3] + 1.669009826152684 weights_num[4]
   + 0.4108501848227875 weights_num[5]
   + 0.0634920634920635 self.weights_cat[1,2] + 0.5 self.weights_cat[1,4]
   + 0.2301587301587301 self.weights_cat[2,1]
   + 0.2936507936507937 self.weights_cat[2,3]
   + 0.1111111111111111 self.weights_cat[3,3]
   + 0.3968253968253969 self.weights_cat[3,4]
   + 0.1031746031746032 self.weights_cat[4,3]
   + 0.246031746031746 self.weights_cat[4,4]
   + 0.3253968253968254 self.weights_cat[4,5]
   + 0.3095238095238095 self.weights_cat[5,2]
   + 0.3571428571428571 self.weights_cat[5,3]
   + 0.1111111111111111 self.weights_cat[5,4] + 0.8015873015873014 bias
   + 0.0952380952380952 tnn_ln_times_numsamples[1,1,4]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,2,4]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,3,4]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,4,4]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,5,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,1,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,2,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,3,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,4,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,5,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,1,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,2,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,3,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,4,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,5,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,1,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,2,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,3,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,4,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,1,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,2,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,3,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,4,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,5,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,6,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,1,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,2,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,3,4]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,4,4]
   = 0.2166685403041464
 cons_first_order_optimality_conditions_num_weights[5]:
   0.632456331086468 weights_num[1] + 0.7483963905196782 weights_num[2]
   + 0.6013263752989778 weights_num[3] + 0.4108501848227875 weights_num[4]
   + 1.86484615050749 weights_num[5]
   + 0.0476190476190476 self.weights_cat[1,2]
   + 0.6849315068493151 self.weights_cat[1,4]
   + 0.3277886497064579 self.weights_cat[2,1]
   + 0.3809523809523809 self.weights_cat[2,3]
   + 0.119047619047619 self.weights_cat[3,3]
   + 0.5658838878016961 self.weights_cat[3,4]
   + 0.0952380952380952 self.weights_cat[4,3]
   + 0.3039791258969341 self.weights_cat[4,4]
   + 0.4761904761904761 self.weights_cat[4,5]
   + 0.3992172211350293 self.weights_cat[5,2]
   + 0.4523809523809523 self.weights_cat[5,3]
   + 0.1428571428571428 self.weights_cat[5,4] + 1.018264840182648 bias
   + 0.0952380952380952 tnn_ln_times_numsamples[1,1,5]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,2,5]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,3,5]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,4,5]
   + 0.0952380952380952 tnn_ln_times_numsamples[1,5,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,1,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,2,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,3,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,4,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,1,5,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,1,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,2,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,3,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,4,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,2,5,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,1,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,2,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,3,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,3,4,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,1,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,2,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,3,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,4,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,5,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,4,6,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,1,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,2,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,3,5]
   + 0.0952380952380952 tcn_lc_times_numsamples[1,5,4,5]
   = 0.264809428421598
 cons_first_order_optimality_conditions_cat_weights[1,1]:
   1.322448979591837 self.weights_cat[1,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,1,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,1,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,1,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,1,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,1,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,1,1] = 0
 cons_first_order_optimality_conditions_cat_weights[1,2]:
   0.0529100529100529 weights_num[1] + 0.0746031746031746 weights_num[2]
   + 0.0952380952380952 weights_num[3] + 0.0634920634920635 weights_num[4]
   + 0.0476190476190476 weights_num[5]
   + 1.417687074829932 self.weights_cat[1,2]
   + 0.0952380952380952 self.weights_cat[2,3]
   + 0.0952380952380952 self.weights_cat[4,4]
   + 0.0952380952380952 self.weights_cat[5,2] + 0.0952380952380952 bias
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,1,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,1,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,1,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,1,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,1,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,1,2]
   = 0.0271258621487756
 cons_first_order_optimality_conditions_cat_weights[1,3]:
   1.322448979591837 self.weights_cat[1,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,1,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,1,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,1,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,1,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,1,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,1,3] = 0
 cons_first_order_optimality_conditions_cat_weights[1,4]:
   0.8253968253968254 weights_num[1] + 0.9984126984126984 weights_num[2]
   + 0.7301587301587301 weights_num[3] + 0.5 weights_num[4]
   + 0.6849315068493149 weights_num[5]
   + 2.560544217687076 self.weights_cat[1,4]
   + 0.4761904761904762 self.weights_cat[2,1]
   + 0.5714285714285714 self.weights_cat[2,3]
   + 0.0952380952380952 self.weights_cat[3,3]
   + 0.9523809523809523 self.weights_cat[3,4]
   + 0.0952380952380952 self.weights_cat[4,3]
   + 0.3809523809523809 self.weights_cat[4,4]
   + 0.7619047619047619 self.weights_cat[4,5]
   + 0.3809523809523809 self.weights_cat[5,2]
   + 0.6666666666666666 self.weights_cat[5,3]
   + 0.1904761904761905 self.weights_cat[5,4] + 1.238095238095238 bias
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,1,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,1,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,1,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,1,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,1,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,1,4]
   = 0.3363157233453025
 cons_first_order_optimality_conditions_cat_weights[1,5]:
   1.322448979591837 self.weights_cat[1,5]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,1,5]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,1,5]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,1,5]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,1,5]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,1,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,1,5] = 0
 cons_first_order_optimality_conditions_cat_weights[3,1]:
   1.322448979591837 self.weights_cat[3,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,3,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,3,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,3,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,3,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,3,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,3,1] = 0
 cons_first_order_optimality_conditions_cat_weights[3,2]:
   1.322448979591837 self.weights_cat[3,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,3,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,3,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,3,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,3,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,3,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,3,2] = 0
 cons_first_order_optimality_conditions_cat_weights[3,3]:
   0.1693121693121693 weights_num[1] + 0.246031746031746 weights_num[2]
   + 0.1587301587301587 weights_num[3] + 0.1111111111111111 weights_num[4]
   + 0.119047619047619 weights_num[5]
   + 0.0952380952380952 self.weights_cat[1,4]
   + 0.1904761904761905 self.weights_cat[2,1]
   + 1.608163265306123 self.weights_cat[3,3]
   + 0.0952380952380952 self.weights_cat[4,4]
   + 0.1904761904761905 self.weights_cat[5,3]
   + 0.0952380952380952 self.weights_cat[5,4] + 0.3809523809523809 bias
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,3,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,3,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,3,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,3,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,3,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,3,3]
   = 0.0704135007703956
 cons_first_order_optimality_conditions_cat_weights[3,4]:
   0.708994708994709 weights_num[1] + 0.9333333333333335 weights_num[2]
   + 0.6031746031746031 weights_num[3] + 0.3968253968253967 weights_num[4]
   + 0.565883887801696 weights_num[5]
   + 0.9523809523809523 self.weights_cat[1,4]
   + 0.3809523809523809 self.weights_cat[2,1]
   + 0.4761904761904762 self.weights_cat[2,3]
   + 2.370068027210884 self.weights_cat[3,4]
   + 0.1904761904761905 self.weights_cat[4,4]
   + 0.8571428571428571 self.weights_cat[4,5]
   + 0.2857142857142857 self.weights_cat[5,2]
   + 0.5714285714285714 self.weights_cat[5,3]
   + 0.1904761904761905 self.weights_cat[5,4] + 1.047619047619047 bias
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,3,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,3,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,3,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,3,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,3,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,3,4]
   = 0.2743798811011697
 cons_first_order_optimality_conditions_cat_weights[4,1]:
   1.322448979591837 self.weights_cat[4,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,4,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,4,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,4,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,4,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,4,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,4,1] = 0
 cons_first_order_optimality_conditions_cat_weights[4,2]:
   1.322448979591837 self.weights_cat[4,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,4,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,4,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,4,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,4,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,4,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,4,2] = 0
 cons_first_order_optimality_conditions_cat_weights[4,3]:
   0.1375661375661376 weights_num[1] + 0.080952380952381 weights_num[2]
   + 0.126984126984127 weights_num[3] + 0.1031746031746032 weights_num[4]
   + 0.0952380952380952 weights_num[5]
   + 0.0952380952380952 self.weights_cat[1,4]
   + 0.0952380952380952 self.weights_cat[2,3]
   + 1.512925170068027 self.weights_cat[4,3]
   + 0.1904761904761905 self.weights_cat[5,2] + 0.1904761904761905 bias
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,4,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,4,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,4,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,4,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,4,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,4,3]
   = 0.0680196533550234
 cons_first_order_optimality_conditions_cat_weights[4,4]:
   0.3068783068783069 weights_num[1] + 0.3777777777777777 weights_num[2]
   + 0.3492063492063492 weights_num[3] + 0.246031746031746 weights_num[4]
   + 0.3039791258969341 weights_num[5]
   + 0.0952380952380952 self.weights_cat[1,2]
   + 0.3809523809523809 self.weights_cat[1,4]
   + 0.2857142857142857 self.weights_cat[2,1]
   + 0.0952380952380952 self.weights_cat[2,3]
   + 0.0952380952380952 self.weights_cat[3,3]
   + 0.1904761904761905 self.weights_cat[3,4]
   + 1.893877551020408 self.weights_cat[4,4]
   + 0.2857142857142857 self.weights_cat[5,2]
   + 0.2857142857142857 self.weights_cat[5,3] + 0.5714285714285714 bias
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,4,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,4,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,4,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,4,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,4,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,4,4]
   = 0.1198378532082184
 cons_first_order_optimality_conditions_cat_weights[4,5]:
   0.5925925925925926 weights_num[1] + 0.761904761904762 weights_num[2]
   + 0.5079365079365079 weights_num[3] + 0.3253968253968254 weights_num[4]
   + 0.4761904761904762 weights_num[5]
   + 0.7619047619047619 self.weights_cat[1,4]
   + 0.2857142857142857 self.weights_cat[2,1]
   + 0.4761904761904762 self.weights_cat[2,3]
   + 0.8571428571428571 self.weights_cat[3,4]
   + 2.179591836734694 self.weights_cat[4,5]
   + 0.1904761904761905 self.weights_cat[5,2]
   + 0.4761904761904762 self.weights_cat[5,3]
   + 0.1904761904761905 self.weights_cat[5,4] + 0.8571428571428571 bias
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,4,5]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,4,5]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,4,5]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,4,5]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,4,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,4,5]
   = 0.2344780156195237
 cons_first_order_optimality_conditions_cat_weights[4,6]:
   1.322448979591837 self.weights_cat[4,6]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,4,6]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,4,6]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,4,6]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,4,6]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,4,6]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,4,6] = 0
 cons_first_order_optimality_conditions_cat_weights[5,1]:
   1.322448979591837 self.weights_cat[5,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,5,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,5,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,5,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,5,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,5,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,5,1] = 0
 cons_first_order_optimality_conditions_cat_weights[5,2]:
   0.4338624338624338 weights_num[1] + 0.4936507936507936 weights_num[2]
   + 0.4126984126984127 weights_num[3] + 0.3095238095238095 weights_num[4]
   + 0.3992172211350293 weights_num[5]
   + 0.0952380952380952 self.weights_cat[1,2]
   + 0.3809523809523809 self.weights_cat[1,4]
   + 0.2857142857142857 self.weights_cat[2,1]
   + 0.2857142857142857 self.weights_cat[2,3]
   + 0.2857142857142857 self.weights_cat[3,4]
   + 0.1904761904761905 self.weights_cat[4,3]
   + 0.2857142857142857 self.weights_cat[4,4]
   + 0.1904761904761905 self.weights_cat[4,5]
   + 2.084353741496599 self.weights_cat[5,2] + 0.7619047619047619 bias
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,5,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,5,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,5,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,5,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,5,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,5,2]
   = 0.1808478981093895
 cons_first_order_optimality_conditions_cat_weights[5,3]:
   0.5925925925925926 weights_num[1] + 0.7111111111111112 weights_num[2]
   + 0.5396825396825397 weights_num[3] + 0.3571428571428571 weights_num[4]
   + 0.4523809523809523 weights_num[5]
   + 0.6666666666666666 self.weights_cat[1,4]
   + 0.3809523809523809 self.weights_cat[2,1]
   + 0.1904761904761905 self.weights_cat[2,3]
   + 0.1904761904761905 self.weights_cat[3,3]
   + 0.5714285714285714 self.weights_cat[3,4]
   + 0.2857142857142857 self.weights_cat[4,4]
   + 0.4761904761904762 self.weights_cat[4,5]
   + 2.179591836734695 self.weights_cat[5,3] + 0.9523809523809523 bias
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,5,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,5,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,5,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,5,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,5,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,5,3]
   = 0.2493039987832378
 cons_first_order_optimality_conditions_cat_weights[5,4]:
   0.1798941798941799 weights_num[1] + 0.2603174603174603 weights_num[2]
   + 0.1904761904761905 weights_num[3] + 0.1111111111111111 weights_num[4]
   + 0.1428571428571428 weights_num[5]
   + 0.1904761904761905 self.weights_cat[1,4]
   + 0.1904761904761905 self.weights_cat[2,3]
   + 0.0952380952380952 self.weights_cat[3,3]
   + 0.1904761904761905 self.weights_cat[3,4]
   + 0.1904761904761905 self.weights_cat[4,5]
   + 1.608163265306122 self.weights_cat[5,4] + 0.2857142857142857 bias
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,5,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,5,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,5,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,5,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,5,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,5,4]
   = 0.0715905859635897
 cons_first_order_optimality_conditions_bias:
   1.206349206349206 weights_num[1] + 1.465079365079365 weights_num[2]
   + 1.142857142857143 weights_num[3] + 0.7777777777777777 weights_num[4]
   + 0.9944553163731249 weights_num[5]
   + 0.0952380952380952 self.weights_cat[1,2]
   + 1.238095238095238 self.weights_cat[1,4]
   + 0.6666666666666666 self.weights_cat[2,1]
   + 0.6666666666666666 self.weights_cat[2,3]
   + 0.2857142857142857 self.weights_cat[3,3]
   + 1.047619047619047 self.weights_cat[3,4]
   + 0.1904761904761905 self.weights_cat[4,3]
   + 0.5714285714285714 self.weights_cat[4,4]
   + 0.8571428571428571 self.weights_cat[4,5]
   + 0.7619047619047619 self.weights_cat[5,2]
   + 0.8571428571428571 self.weights_cat[5,3]
   + 0.2857142857142857 self.weights_cat[5,4] + 2 bias
   + 0.0952380952380952 ln_numweight_times_numsample[1,1]
   + 0.0952380952380952 ln_numweight_times_numsample[1,2]
   + 0.0952380952380952 ln_numweight_times_numsample[1,3]
   + 0.0952380952380952 ln_numweight_times_numsample[1,4]
   + 0.0952380952380952 ln_numweight_times_numsample[1,5]
   + 0.0952380952380952 lc_catweight_times_catsample[1,1,1]
   + 0.0952380952380952 lc_catweight_times_catsample[1,1,2]
   + 0.0952380952380952 lc_catweight_times_catsample[1,1,3]
   + 0.0952380952380952 lc_catweight_times_catsample[1,1,4]
   + 0.0952380952380952 lc_catweight_times_catsample[1,1,5]
   + 0.0952380952380952 lc_catweight_times_catsample[1,2,1]
   + 0.0952380952380952 lc_catweight_times_catsample[1,2,2]
   + 0.0952380952380952 lc_catweight_times_catsample[1,2,3]
   + 0.0952380952380952 lc_catweight_times_catsample[1,2,4]
   + 0.0952380952380952 lc_catweight_times_catsample[1,2,5]
   + 0.0952380952380952 lc_catweight_times_catsample[1,3,1]
   + 0.0952380952380952 lc_catweight_times_catsample[1,3,2]
   + 0.0952380952380952 lc_catweight_times_catsample[1,3,3]
   + 0.0952380952380952 lc_catweight_times_catsample[1,3,4]
   + 0.0952380952380952 lc_catweight_times_catsample[1,4,1]
   + 0.0952380952380952 lc_catweight_times_catsample[1,4,2]
   + 0.0952380952380952 lc_catweight_times_catsample[1,4,3]
   + 0.0952380952380952 lc_catweight_times_catsample[1,4,4]
   + 0.0952380952380952 lc_catweight_times_catsample[1,4,5]
   + 0.0952380952380952 lc_catweight_times_catsample[1,4,6]
   + 0.0952380952380952 lc_catweight_times_catsample[1,5,1]
   + 0.0952380952380952 lc_catweight_times_catsample[1,5,2]
   + 0.0952380952380952 lc_catweight_times_catsample[1,5,3]
   + 0.0952380952380952 lc_catweight_times_catsample[1,5,4]
   = 0.5017424828562171
 cons_tnn[1,1,1]: - tnn_ln_times_numsamples[1,1,1]
   + 0.5555555555555556 ln_numweight_times_numsample[1,1] = 0
 cons_tnn[1,1,2]: - tnn_ln_times_numsamples[1,1,2]
   + 0.95 ln_numweight_times_numsample[1,1] = 0
 cons_tnn[1,1,3]: - tnn_ln_times_numsamples[1,1,3]
   + 0.3333333333333333 ln_numweight_times_numsample[1,1] = 0
 cons_tnn[1,1,4]: - tnn_ln_times_numsamples[1,1,4]
   + 0.25 ln_numweight_times_numsample[1,1] = 0
 cons_tnn[1,1,5]: - tnn_ln_times_numsamples[1,1,5]
   + 0.25 ln_numweight_times_numsample[1,1] = 0
 cons_tnn[1,2,1]: - tnn_ln_times_numsamples[1,2,1]
   + 0.5555555555555556 ln_numweight_times_numsample[1,2] = 0
 cons_tnn[1,2,2]: - tnn_ln_times_numsamples[1,2,2]
   + 0.95 ln_numweight_times_numsample[1,2] = 0
 cons_tnn[1,2,3]: - tnn_ln_times_numsamples[1,2,3]
   + 0.3333333333333333 ln_numweight_times_numsample[1,2] = 0
 cons_tnn[1,2,4]: - tnn_ln_times_numsamples[1,2,4]
   + 0.25 ln_numweight_times_numsample[1,2] = 0
 cons_tnn[1,2,5]: - tnn_ln_times_numsamples[1,2,5]
   + 0.25 ln_numweight_times_numsample[1,2] = 0
 cons_tnn[1,3,1]: - tnn_ln_times_numsamples[1,3,1]
   + 0.5555555555555556 ln_numweight_times_numsample[1,3] = 0
 cons_tnn[1,3,2]: - tnn_ln_times_numsamples[1,3,2]
   + 0.95 ln_numweight_times_numsample[1,3] = 0
 cons_tnn[1,3,3]: - tnn_ln_times_numsamples[1,3,3]
   + 0.3333333333333333 ln_numweight_times_numsample[1,3] = 0
 cons_tnn[1,3,4]: - tnn_ln_times_numsamples[1,3,4]
   + 0.25 ln_numweight_times_numsample[1,3] = 0
 cons_tnn[1,3,5]: - tnn_ln_times_numsamples[1,3,5]
   + 0.25 ln_numweight_times_numsample[1,3] = 0
 cons_tnn[1,4,1]: - tnn_ln_times_numsamples[1,4,1]
   + 0.5555555555555556 ln_numweight_times_numsample[1,4] = 0
 cons_tnn[1,4,2]: - tnn_ln_times_numsamples[1,4,2]
   + 0.95 ln_numweight_times_numsample[1,4] = 0
 cons_tnn[1,4,3]: - tnn_ln_times_numsamples[1,4,3]
   + 0.3333333333333333 ln_numweight_times_numsample[1,4] = 0
 cons_tnn[1,4,4]: - tnn_ln_times_numsamples[1,4,4]
   + 0.25 ln_numweight_times_numsample[1,4] = 0
 cons_tnn[1,4,5]: - tnn_ln_times_numsamples[1,4,5]
   + 0.25 ln_numweight_times_numsample[1,4] = 0
 cons_tnn[1,5,1]: - tnn_ln_times_numsamples[1,5,1]
   + 0.5555555555555556 ln_numweight_times_numsample[1,5] = 0
 cons_tnn[1,5,2]: - tnn_ln_times_numsamples[1,5,2]
   + 0.95 ln_numweight_times_numsample[1,5] = 0
 cons_tnn[1,5,3]: - tnn_ln_times_numsamples[1,5,3]
   + 0.3333333333333333 ln_numweight_times_numsample[1,5] = 0
 cons_tnn[1,5,4]: - tnn_ln_times_numsamples[1,5,4]
   + 0.25 ln_numweight_times_numsample[1,5] = 0
 cons_tnn[1,5,5]: - tnn_ln_times_numsamples[1,5,5]
   + 0.25 ln_numweight_times_numsample[1,5] = 0
 cons_tcn[1,1,1,1]: - tcn_lc_times_numsamples[1,1,1,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,1,1] = 0
 cons_tcn[1,1,1,2]: - tcn_lc_times_numsamples[1,1,1,2]
   + 0.95 lc_catweight_times_catsample[1,1,1] = 0
 cons_tcn[1,1,1,3]: - tcn_lc_times_numsamples[1,1,1,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,1,1] = 0
 cons_tcn[1,1,1,4]: - tcn_lc_times_numsamples[1,1,1,4]
   + 0.25 lc_catweight_times_catsample[1,1,1] = 0
 cons_tcn[1,1,1,5]: - tcn_lc_times_numsamples[1,1,1,5]
   + 0.25 lc_catweight_times_catsample[1,1,1] = 0
 cons_tcn[1,1,2,1]: - tcn_lc_times_numsamples[1,1,2,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,1,2] = 0
 cons_tcn[1,1,2,2]: - tcn_lc_times_numsamples[1,1,2,2]
   + 0.95 lc_catweight_times_catsample[1,1,2] = 0
 cons_tcn[1,1,2,3]: - tcn_lc_times_numsamples[1,1,2,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,1,2] = 0
 cons_tcn[1,1,2,4]: - tcn_lc_times_numsamples[1,1,2,4]
   + 0.25 lc_catweight_times_catsample[1,1,2] = 0
 cons_tcn[1,1,2,5]: - tcn_lc_times_numsamples[1,1,2,5]
   + 0.25 lc_catweight_times_catsample[1,1,2] = 0
 cons_tcn[1,1,3,1]: - tcn_lc_times_numsamples[1,1,3,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,1,3] = 0
 cons_tcn[1,1,3,2]: - tcn_lc_times_numsamples[1,1,3,2]
   + 0.95 lc_catweight_times_catsample[1,1,3] = 0
 cons_tcn[1,1,3,3]: - tcn_lc_times_numsamples[1,1,3,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,1,3] = 0
 cons_tcn[1,1,3,4]: - tcn_lc_times_numsamples[1,1,3,4]
   + 0.25 lc_catweight_times_catsample[1,1,3] = 0
 cons_tcn[1,1,3,5]: - tcn_lc_times_numsamples[1,1,3,5]
   + 0.25 lc_catweight_times_catsample[1,1,3] = 0
 cons_tcn[1,1,4,1]: - tcn_lc_times_numsamples[1,1,4,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,1,4] = 0
 cons_tcn[1,1,4,2]: - tcn_lc_times_numsamples[1,1,4,2]
   + 0.95 lc_catweight_times_catsample[1,1,4] = 0
 cons_tcn[1,1,4,3]: - tcn_lc_times_numsamples[1,1,4,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,1,4] = 0
 cons_tcn[1,1,4,4]: - tcn_lc_times_numsamples[1,1,4,4]
   + 0.25 lc_catweight_times_catsample[1,1,4] = 0
 cons_tcn[1,1,4,5]: - tcn_lc_times_numsamples[1,1,4,5]
   + 0.25 lc_catweight_times_catsample[1,1,4] = 0
 cons_tcn[1,1,5,1]: - tcn_lc_times_numsamples[1,1,5,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,1,5] = 0
 cons_tcn[1,1,5,2]: - tcn_lc_times_numsamples[1,1,5,2]
   + 0.95 lc_catweight_times_catsample[1,1,5] = 0
 cons_tcn[1,1,5,3]: - tcn_lc_times_numsamples[1,1,5,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,1,5] = 0
 cons_tcn[1,1,5,4]: - tcn_lc_times_numsamples[1,1,5,4]
   + 0.25 lc_catweight_times_catsample[1,1,5] = 0
 cons_tcn[1,1,5,5]: - tcn_lc_times_numsamples[1,1,5,5]
   + 0.25 lc_catweight_times_catsample[1,1,5] = 0
 cons_tcn[1,2,1,1]: - tcn_lc_times_numsamples[1,2,1,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,2,1] = 0
 cons_tcn[1,2,1,2]: - tcn_lc_times_numsamples[1,2,1,2]
   + 0.95 lc_catweight_times_catsample[1,2,1] = 0
 cons_tcn[1,2,1,3]: - tcn_lc_times_numsamples[1,2,1,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,2,1] = 0
 cons_tcn[1,2,1,4]: - tcn_lc_times_numsamples[1,2,1,4]
   + 0.25 lc_catweight_times_catsample[1,2,1] = 0
 cons_tcn[1,2,1,5]: - tcn_lc_times_numsamples[1,2,1,5]
   + 0.25 lc_catweight_times_catsample[1,2,1] = 0
 cons_tcn[1,2,2,1]: - tcn_lc_times_numsamples[1,2,2,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,2,2] = 0
 cons_tcn[1,2,2,2]: - tcn_lc_times_numsamples[1,2,2,2]
   + 0.95 lc_catweight_times_catsample[1,2,2] = 0
 cons_tcn[1,2,2,3]: - tcn_lc_times_numsamples[1,2,2,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,2,2] = 0
 cons_tcn[1,2,2,4]: - tcn_lc_times_numsamples[1,2,2,4]
   + 0.25 lc_catweight_times_catsample[1,2,2] = 0
 cons_tcn[1,2,2,5]: - tcn_lc_times_numsamples[1,2,2,5]
   + 0.25 lc_catweight_times_catsample[1,2,2] = 0
 cons_tcn[1,2,3,1]: - tcn_lc_times_numsamples[1,2,3,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,2,3] = 0
 cons_tcn[1,2,3,2]: - tcn_lc_times_numsamples[1,2,3,2]
   + 0.95 lc_catweight_times_catsample[1,2,3] = 0
 cons_tcn[1,2,3,3]: - tcn_lc_times_numsamples[1,2,3,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,2,3] = 0
 cons_tcn[1,2,3,4]: - tcn_lc_times_numsamples[1,2,3,4]
   + 0.25 lc_catweight_times_catsample[1,2,3] = 0
 cons_tcn[1,2,3,5]: - tcn_lc_times_numsamples[1,2,3,5]
   + 0.25 lc_catweight_times_catsample[1,2,3] = 0
 cons_tcn[1,2,4,1]: - tcn_lc_times_numsamples[1,2,4,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,2,4] = 0
 cons_tcn[1,2,4,2]: - tcn_lc_times_numsamples[1,2,4,2]
   + 0.95 lc_catweight_times_catsample[1,2,4] = 0
 cons_tcn[1,2,4,3]: - tcn_lc_times_numsamples[1,2,4,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,2,4] = 0
 cons_tcn[1,2,4,4]: - tcn_lc_times_numsamples[1,2,4,4]
   + 0.25 lc_catweight_times_catsample[1,2,4] = 0
 cons_tcn[1,2,4,5]: - tcn_lc_times_numsamples[1,2,4,5]
   + 0.25 lc_catweight_times_catsample[1,2,4] = 0
 cons_tcn[1,2,5,1]: - tcn_lc_times_numsamples[1,2,5,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,2,5] = 0
 cons_tcn[1,2,5,2]: - tcn_lc_times_numsamples[1,2,5,2]
   + 0.95 lc_catweight_times_catsample[1,2,5] = 0
 cons_tcn[1,2,5,3]: - tcn_lc_times_numsamples[1,2,5,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,2,5] = 0
 cons_tcn[1,2,5,4]: - tcn_lc_times_numsamples[1,2,5,4]
   + 0.25 lc_catweight_times_catsample[1,2,5] = 0
 cons_tcn[1,2,5,5]: - tcn_lc_times_numsamples[1,2,5,5]
   + 0.25 lc_catweight_times_catsample[1,2,5] = 0
 cons_tcn[1,3,1,1]: - tcn_lc_times_numsamples[1,3,1,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,3,1] = 0
 cons_tcn[1,3,1,2]: - tcn_lc_times_numsamples[1,3,1,2]
   + 0.95 lc_catweight_times_catsample[1,3,1] = 0
 cons_tcn[1,3,1,3]: - tcn_lc_times_numsamples[1,3,1,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,3,1] = 0
 cons_tcn[1,3,1,4]: - tcn_lc_times_numsamples[1,3,1,4]
   + 0.25 lc_catweight_times_catsample[1,3,1] = 0
 cons_tcn[1,3,1,5]: - tcn_lc_times_numsamples[1,3,1,5]
   + 0.25 lc_catweight_times_catsample[1,3,1] = 0
 cons_tcn[1,3,2,1]: - tcn_lc_times_numsamples[1,3,2,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,3,2] = 0
 cons_tcn[1,3,2,2]: - tcn_lc_times_numsamples[1,3,2,2]
   + 0.95 lc_catweight_times_catsample[1,3,2] = 0
 cons_tcn[1,3,2,3]: - tcn_lc_times_numsamples[1,3,2,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,3,2] = 0
 cons_tcn[1,3,2,4]: - tcn_lc_times_numsamples[1,3,2,4]
   + 0.25 lc_catweight_times_catsample[1,3,2] = 0
 cons_tcn[1,3,2,5]: - tcn_lc_times_numsamples[1,3,2,5]
   + 0.25 lc_catweight_times_catsample[1,3,2] = 0
 cons_tcn[1,3,3,1]: - tcn_lc_times_numsamples[1,3,3,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,3,3] = 0
 cons_tcn[1,3,3,2]: - tcn_lc_times_numsamples[1,3,3,2]
   + 0.95 lc_catweight_times_catsample[1,3,3] = 0
 cons_tcn[1,3,3,3]: - tcn_lc_times_numsamples[1,3,3,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,3,3] = 0
 cons_tcn[1,3,3,4]: - tcn_lc_times_numsamples[1,3,3,4]
   + 0.25 lc_catweight_times_catsample[1,3,3] = 0
 cons_tcn[1,3,3,5]: - tcn_lc_times_numsamples[1,3,3,5]
   + 0.25 lc_catweight_times_catsample[1,3,3] = 0
 cons_tcn[1,3,4,1]: - tcn_lc_times_numsamples[1,3,4,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,3,4] = 0
 cons_tcn[1,3,4,2]: - tcn_lc_times_numsamples[1,3,4,2]
   + 0.95 lc_catweight_times_catsample[1,3,4] = 0
 cons_tcn[1,3,4,3]: - tcn_lc_times_numsamples[1,3,4,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,3,4] = 0
 cons_tcn[1,3,4,4]: - tcn_lc_times_numsamples[1,3,4,4]
   + 0.25 lc_catweight_times_catsample[1,3,4] = 0
 cons_tcn[1,3,4,5]: - tcn_lc_times_numsamples[1,3,4,5]
   + 0.25 lc_catweight_times_catsample[1,3,4] = 0
 cons_tcn[1,4,1,1]: - tcn_lc_times_numsamples[1,4,1,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,4,1] = 0
 cons_tcn[1,4,1,2]: - tcn_lc_times_numsamples[1,4,1,2]
   + 0.95 lc_catweight_times_catsample[1,4,1] = 0
 cons_tcn[1,4,1,3]: - tcn_lc_times_numsamples[1,4,1,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,4,1] = 0
 cons_tcn[1,4,1,4]: - tcn_lc_times_numsamples[1,4,1,4]
   + 0.25 lc_catweight_times_catsample[1,4,1] = 0
 cons_tcn[1,4,1,5]: - tcn_lc_times_numsamples[1,4,1,5]
   + 0.25 lc_catweight_times_catsample[1,4,1] = 0
 cons_tcn[1,4,2,1]: - tcn_lc_times_numsamples[1,4,2,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,4,2] = 0
 cons_tcn[1,4,2,2]: - tcn_lc_times_numsamples[1,4,2,2]
   + 0.95 lc_catweight_times_catsample[1,4,2] = 0
 cons_tcn[1,4,2,3]: - tcn_lc_times_numsamples[1,4,2,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,4,2] = 0
 cons_tcn[1,4,2,4]: - tcn_lc_times_numsamples[1,4,2,4]
   + 0.25 lc_catweight_times_catsample[1,4,2] = 0
 cons_tcn[1,4,2,5]: - tcn_lc_times_numsamples[1,4,2,5]
   + 0.25 lc_catweight_times_catsample[1,4,2] = 0
 cons_tcn[1,4,3,1]: - tcn_lc_times_numsamples[1,4,3,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,4,3] = 0
 cons_tcn[1,4,3,2]: - tcn_lc_times_numsamples[1,4,3,2]
   + 0.95 lc_catweight_times_catsample[1,4,3] = 0
 cons_tcn[1,4,3,3]: - tcn_lc_times_numsamples[1,4,3,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,4,3] = 0
 cons_tcn[1,4,3,4]: - tcn_lc_times_numsamples[1,4,3,4]
   + 0.25 lc_catweight_times_catsample[1,4,3] = 0
 cons_tcn[1,4,3,5]: - tcn_lc_times_numsamples[1,4,3,5]
   + 0.25 lc_catweight_times_catsample[1,4,3] = 0
 cons_tcn[1,4,4,1]: - tcn_lc_times_numsamples[1,4,4,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,4,4] = 0
 cons_tcn[1,4,4,2]: - tcn_lc_times_numsamples[1,4,4,2]
   + 0.95 lc_catweight_times_catsample[1,4,4] = 0
 cons_tcn[1,4,4,3]: - tcn_lc_times_numsamples[1,4,4,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,4,4] = 0
 cons_tcn[1,4,4,4]: - tcn_lc_times_numsamples[1,4,4,4]
   + 0.25 lc_catweight_times_catsample[1,4,4] = 0
 cons_tcn[1,4,4,5]: - tcn_lc_times_numsamples[1,4,4,5]
   + 0.25 lc_catweight_times_catsample[1,4,4] = 0
 cons_tcn[1,4,5,1]: - tcn_lc_times_numsamples[1,4,5,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,4,5] = 0
 cons_tcn[1,4,5,2]: - tcn_lc_times_numsamples[1,4,5,2]
   + 0.95 lc_catweight_times_catsample[1,4,5] = 0
 cons_tcn[1,4,5,3]: - tcn_lc_times_numsamples[1,4,5,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,4,5] = 0
 cons_tcn[1,4,5,4]: - tcn_lc_times_numsamples[1,4,5,4]
   + 0.25 lc_catweight_times_catsample[1,4,5] = 0
 cons_tcn[1,4,5,5]: - tcn_lc_times_numsamples[1,4,5,5]
   + 0.25 lc_catweight_times_catsample[1,4,5] = 0
 cons_tcn[1,4,6,1]: - tcn_lc_times_numsamples[1,4,6,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,4,6] = 0
 cons_tcn[1,4,6,2]: - tcn_lc_times_numsamples[1,4,6,2]
   + 0.95 lc_catweight_times_catsample[1,4,6] = 0
 cons_tcn[1,4,6,3]: - tcn_lc_times_numsamples[1,4,6,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,4,6] = 0
 cons_tcn[1,4,6,4]: - tcn_lc_times_numsamples[1,4,6,4]
   + 0.25 lc_catweight_times_catsample[1,4,6] = 0
 cons_tcn[1,4,6,5]: - tcn_lc_times_numsamples[1,4,6,5]
   + 0.25 lc_catweight_times_catsample[1,4,6] = 0
 cons_tcn[1,5,1,1]: - tcn_lc_times_numsamples[1,5,1,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,5,1] = 0
 cons_tcn[1,5,1,2]: - tcn_lc_times_numsamples[1,5,1,2]
   + 0.95 lc_catweight_times_catsample[1,5,1] = 0
 cons_tcn[1,5,1,3]: - tcn_lc_times_numsamples[1,5,1,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,5,1] = 0
 cons_tcn[1,5,1,4]: - tcn_lc_times_numsamples[1,5,1,4]
   + 0.25 lc_catweight_times_catsample[1,5,1] = 0
 cons_tcn[1,5,1,5]: - tcn_lc_times_numsamples[1,5,1,5]
   + 0.25 lc_catweight_times_catsample[1,5,1] = 0
 cons_tcn[1,5,2,1]: - tcn_lc_times_numsamples[1,5,2,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,5,2] = 0
 cons_tcn[1,5,2,2]: - tcn_lc_times_numsamples[1,5,2,2]
   + 0.95 lc_catweight_times_catsample[1,5,2] = 0
 cons_tcn[1,5,2,3]: - tcn_lc_times_numsamples[1,5,2,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,5,2] = 0
 cons_tcn[1,5,2,4]: - tcn_lc_times_numsamples[1,5,2,4]
   + 0.25 lc_catweight_times_catsample[1,5,2] = 0
 cons_tcn[1,5,2,5]: - tcn_lc_times_numsamples[1,5,2,5]
   + 0.25 lc_catweight_times_catsample[1,5,2] = 0
 cons_tcn[1,5,3,1]: - tcn_lc_times_numsamples[1,5,3,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,5,3] = 0
 cons_tcn[1,5,3,2]: - tcn_lc_times_numsamples[1,5,3,2]
   + 0.95 lc_catweight_times_catsample[1,5,3] = 0
 cons_tcn[1,5,3,3]: - tcn_lc_times_numsamples[1,5,3,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,5,3] = 0
 cons_tcn[1,5,3,4]: - tcn_lc_times_numsamples[1,5,3,4]
   + 0.25 lc_catweight_times_catsample[1,5,3] = 0
 cons_tcn[1,5,3,5]: - tcn_lc_times_numsamples[1,5,3,5]
   + 0.25 lc_catweight_times_catsample[1,5,3] = 0
 cons_tcn[1,5,4,1]: - tcn_lc_times_numsamples[1,5,4,1]
   + 0.5555555555555556 lc_catweight_times_catsample[1,5,4] = 0
 cons_tcn[1,5,4,2]: - tcn_lc_times_numsamples[1,5,4,2]
   + 0.95 lc_catweight_times_catsample[1,5,4] = 0
 cons_tcn[1,5,4,3]: - tcn_lc_times_numsamples[1,5,4,3]
   + 0.3333333333333333 lc_catweight_times_catsample[1,5,4] = 0
 cons_tcn[1,5,4,4]: - tcn_lc_times_numsamples[1,5,4,4]
   + 0.25 lc_catweight_times_catsample[1,5,4] = 0
 cons_tcn[1,5,4,5]: - tcn_lc_times_numsamples[1,5,4,5]
   + 0.25 lc_catweight_times_catsample[1,5,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,1,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,1,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,1,3] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,1,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,1,5] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,3,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,3,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,3,3]
   + ln_numweight_times_numsample[1,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,3,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,4,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,4,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,4,3] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,4,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,4,5] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,4,6] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,5,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,5,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,5,3]
   + ln_numweight_times_numsample[1,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,1,5,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,1,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,1,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,1,3] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,1,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,1,5] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,3,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,3,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,3,3]
   + ln_numweight_times_numsample[1,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,3,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,4,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,4,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,4,3] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,4,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,4,5] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,4,6] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,5,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,5,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,5,3]
   + ln_numweight_times_numsample[1,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,2,5,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,1,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,1,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,1,3] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,1,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,1,5] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,3,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,3,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,3,3]
   + ln_numweight_times_numsample[1,3] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,3,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,4,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,4,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,4,3] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,4,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,4,5] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,4,6] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,5,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,5,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,5,3]
   + ln_numweight_times_numsample[1,3] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,3,5,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,1,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,1,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,1,3] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,1,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,1,5] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,3,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,3,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,3,3]
   + ln_numweight_times_numsample[1,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,3,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,4,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,4,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,4,3] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,4,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,4,5] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,4,6] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,5,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,5,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,5,3]
   + ln_numweight_times_numsample[1,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,4,5,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,1,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,1,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,1,3] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,1,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,1,5] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,3,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,3,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,3,3]
   + ln_numweight_times_numsample[1,5] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,3,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,4,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,4,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,4,3] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,4,4] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,4,5] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,4,6] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,5,1] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,5,2] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,5,3]
   + ln_numweight_times_numsample[1,5] = 0
 cons_tnc: - tnc_ln_times_catsamples[1,5,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,3,3]
   + lc_catweight_times_catsample[1,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,5,3]
   + lc_catweight_times_catsample[1,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,1,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,3,3]
   + lc_catweight_times_catsample[1,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,5,3]
   + lc_catweight_times_catsample[1,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,2,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,3,3]
   + lc_catweight_times_catsample[1,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,5,3]
   + lc_catweight_times_catsample[1,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,3,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,3,3]
   + lc_catweight_times_catsample[1,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,5,3]
   + lc_catweight_times_catsample[1,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,4,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,3,3]
   + lc_catweight_times_catsample[1,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,5,3]
   + lc_catweight_times_catsample[1,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,1,5,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,3,3]
   + lc_catweight_times_catsample[1,2,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,5,3]
   + lc_catweight_times_catsample[1,2,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,1,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,3,3]
   + lc_catweight_times_catsample[1,2,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,5,3]
   + lc_catweight_times_catsample[1,2,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,2,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,3,3]
   + lc_catweight_times_catsample[1,2,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,5,3]
   + lc_catweight_times_catsample[1,2,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,3,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,3,3]
   + lc_catweight_times_catsample[1,2,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,5,3]
   + lc_catweight_times_catsample[1,2,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,4,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,3,3]
   + lc_catweight_times_catsample[1,2,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,5,3]
   + lc_catweight_times_catsample[1,2,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,2,5,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,3,3]
   + lc_catweight_times_catsample[1,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,5,3]
   + lc_catweight_times_catsample[1,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,1,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,3,3]
   + lc_catweight_times_catsample[1,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,5,3]
   + lc_catweight_times_catsample[1,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,2,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,3,3]
   + lc_catweight_times_catsample[1,3,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,5,3]
   + lc_catweight_times_catsample[1,3,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,3,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,3,3]
   + lc_catweight_times_catsample[1,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,5,3]
   + lc_catweight_times_catsample[1,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,3,4,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,3,3]
   + lc_catweight_times_catsample[1,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,5,3]
   + lc_catweight_times_catsample[1,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,1,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,3,3]
   + lc_catweight_times_catsample[1,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,5,3]
   + lc_catweight_times_catsample[1,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,2,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,3,3]
   + lc_catweight_times_catsample[1,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,5,3]
   + lc_catweight_times_catsample[1,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,3,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,3,3]
   + lc_catweight_times_catsample[1,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,5,3]
   + lc_catweight_times_catsample[1,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,4,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,3,3]
   + lc_catweight_times_catsample[1,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,5,3]
   + lc_catweight_times_catsample[1,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,5,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,3,3]
   + lc_catweight_times_catsample[1,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,5,3]
   + lc_catweight_times_catsample[1,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,4,6,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,3,3]
   + lc_catweight_times_catsample[1,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,5,3]
   + lc_catweight_times_catsample[1,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,1,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,3,3]
   + lc_catweight_times_catsample[1,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,5,3]
   + lc_catweight_times_catsample[1,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,2,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,3,3]
   + lc_catweight_times_catsample[1,5,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,5,3]
   + lc_catweight_times_catsample[1,5,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,3,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,1,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,1,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,1,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,1,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,1,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,3,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,3,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,3,3]
   + lc_catweight_times_catsample[1,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,3,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,4,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,4,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,4,3] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,4,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,4,5] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,4,6] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,5,1] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,5,2] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,5,3]
   + lc_catweight_times_catsample[1,5,4] = 0
 cons_tcc: - tcc_lc_times_catsample[1,5,4,5,4] = 0
 cons_ln: 0.5555555555555556 weights_num[1]
   - ln_numweight_times_numsample[1,1] = 0
 cons_ln: 0.95 weights_num[2] - ln_numweight_times_numsample[1,2] = 0
 cons_ln: 0.3333333333333333 weights_num[3]
   - ln_numweight_times_numsample[1,3] = 0
 cons_ln: 0.25 weights_num[4] - ln_numweight_times_numsample[1,4] = 0
 cons_ln: 0.25 weights_num[5] - ln_numweight_times_numsample[1,5] = 0
 cons_lc: - lc_catweight_times_catsample[1,1,1] = 0
 cons_lc: - lc_catweight_times_catsample[1,1,2] = 0
 cons_lc: - lc_catweight_times_catsample[1,1,3] = 0
 cons_lc: - lc_catweight_times_catsample[1,1,4] = 0
 cons_lc: - lc_catweight_times_catsample[1,1,5] = 0
 cons_lc: - lc_catweight_times_catsample[1,3,1] = 0
 cons_lc: - lc_catweight_times_catsample[1,3,2] = 0
 cons_lc: self.weights_cat[3,3] - lc_catweight_times_catsample[1,3,3] = 0
 cons_lc: - lc_catweight_times_catsample[1,3,4] = 0
 cons_lc: - lc_catweight_times_catsample[1,4,1] = 0
 cons_lc: - lc_catweight_times_catsample[1,4,2] = 0
 cons_lc: - lc_catweight_times_catsample[1,4,3] = 0
 cons_lc: - lc_catweight_times_catsample[1,4,4] = 0
 cons_lc: - lc_catweight_times_catsample[1,4,5] = 0
 cons_lc: - lc_catweight_times_catsample[1,4,6] = 0
 cons_lc: - lc_catweight_times_catsample[1,5,1] = 0
 cons_lc: - lc_catweight_times_catsample[1,5,2] = 0
 cons_lc: self.weights_cat[5,3] - lc_catweight_times_catsample[1,5,3] = 0
 cons_lc: - lc_catweight_times_catsample[1,5,4] = 0
 cons_first_order_optimality_conditions_pcat_weights[2,1]:
   - 0.0109905370285873 x_poison_cat[1,2,1]
   + 0.3386243386243386 weights_num[1] + 0.4952380952380952 weights_num[2]
   + 0.3174603174603174 weights_num[3] + 0.2301587301587301 weights_num[4]
   + 0.3277886497064579 weights_num[5]
   + 0.4761904761904762 self.weights_cat[1,4]
   + 1.989115646258504 self.weights_cat[2,1]
   + 0.1904761904761905 self.weights_cat[3,3]
   + 0.3809523809523809 self.weights_cat[3,4]
   + 0.2857142857142857 self.weights_cat[4,4]
   + 0.2857142857142857 self.weights_cat[4,5]
   + 0.2857142857142857 self.weights_cat[5,2]
   + 0.3809523809523809 self.weights_cat[5,3] + 0.6666666666666666 bias
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,2,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,2,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,2,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,2,1]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,2,1]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,2,1] + [
   0.0952380952380952 x_poison_cat[1,2,1] * bias ] = 0.1136350110103755
 cons_first_order_optimality_conditions_pcat_weights[2,2]:
   - 0.0109905370285873 x_poison_cat[1,2,2]
   + 1.322448979591837 self.weights_cat[2,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,2,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,2,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,2,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,2,2]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,2,2]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,2,2] + [
   0.0952380952380952 x_poison_cat[1,2,2] * bias ] = 0
 cons_first_order_optimality_conditions_pcat_weights[2,3]:
   - 0.0109905370285873 x_poison_cat[1,2,3]
   + 0.4867724867724867 weights_num[1] + 0.5936507936507935 weights_num[2]
   + 0.4444444444444444 weights_num[3] + 0.2936507936507936 weights_num[4]
   + 0.3809523809523809 weights_num[5]
   + 0.0952380952380952 self.weights_cat[1,2]
   + 0.5714285714285714 self.weights_cat[1,4]
   + 1.989115646258504 self.weights_cat[2,3]
   + 0.4761904761904762 self.weights_cat[3,4]
   + 0.0952380952380952 self.weights_cat[4,3]
   + 0.0952380952380952 self.weights_cat[4,4]
   + 0.4761904761904762 self.weights_cat[4,5]
   + 0.2857142857142857 self.weights_cat[5,2]
   + 0.1904761904761905 self.weights_cat[5,3]
   + 0.1904761904761905 self.weights_cat[5,4] + 0.6666666666666666 bias
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,2,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,2,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,2,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,2,3]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,2,3]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,2,3] + [
   0.0952380952380952 x_poison_cat[1,2,3] * bias ] = 0.2092037481566713
 cons_first_order_optimality_conditions_pcat_weights[2,4]:
   - 0.0109905370285873 x_poison_cat[1,2,4]
   + 1.322448979591837 self.weights_cat[2,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,2,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,2,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,2,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,2,4]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,2,4]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,2,4] + [
   0.0952380952380952 x_poison_cat[1,2,4] * bias ] = 0
 cons_first_order_optimality_conditions_pcat_weights[2,5]:
   - 0.0109905370285873 x_poison_cat[1,2,5]
   + 1.322448979591837 self.weights_cat[2,5]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,1,2,5]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,2,2,5]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,3,2,5]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,4,2,5]
   + 0.0952380952380952 tnc_ln_times_catsamples[1,5,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,1,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,2,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,3,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,4,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,1,5,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,1,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,2,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,3,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,4,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,2,5,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,1,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,2,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,3,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,3,4,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,1,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,2,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,3,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,4,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,5,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,4,6,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,1,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,2,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,3,2,5]
   + 0.0952380952380952 tcc_lc_times_catsample[1,5,4,2,5] + [
   0.0952380952380952 x_poison_cat[1,2,5] * bias ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,1,2,1] + [
   x_poison_cat[1,2,1] * ln_numweight_times_numsample[1,1] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,1,2,2] + [
   x_poison_cat[1,2,2] * ln_numweight_times_numsample[1,1] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,1,2,3] + [
   x_poison_cat[1,2,3] * ln_numweight_times_numsample[1,1] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,1,2,4] + [
   x_poison_cat[1,2,4] * ln_numweight_times_numsample[1,1] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,1,2,5] + [
   x_poison_cat[1,2,5] * ln_numweight_times_numsample[1,1] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,2,2,1] + [
   x_poison_cat[1,2,1] * ln_numweight_times_numsample[1,2] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,2,2,2] + [
   x_poison_cat[1,2,2] * ln_numweight_times_numsample[1,2] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,2,2,3] + [
   x_poison_cat[1,2,3] * ln_numweight_times_numsample[1,2] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,2,2,4] + [
   x_poison_cat[1,2,4] * ln_numweight_times_numsample[1,2] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,2,2,5] + [
   x_poison_cat[1,2,5] * ln_numweight_times_numsample[1,2] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,3,2,1] + [
   x_poison_cat[1,2,1] * ln_numweight_times_numsample[1,3] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,3,2,2] + [
   x_poison_cat[1,2,2] * ln_numweight_times_numsample[1,3] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,3,2,3] + [
   x_poison_cat[1,2,3] * ln_numweight_times_numsample[1,3] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,3,2,4] + [
   x_poison_cat[1,2,4] * ln_numweight_times_numsample[1,3] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,3,2,5] + [
   x_poison_cat[1,2,5] * ln_numweight_times_numsample[1,3] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,4,2,1] + [
   x_poison_cat[1,2,1] * ln_numweight_times_numsample[1,4] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,4,2,2] + [
   x_poison_cat[1,2,2] * ln_numweight_times_numsample[1,4] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,4,2,3] + [
   x_poison_cat[1,2,3] * ln_numweight_times_numsample[1,4] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,4,2,4] + [
   x_poison_cat[1,2,4] * ln_numweight_times_numsample[1,4] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,4,2,5] + [
   x_poison_cat[1,2,5] * ln_numweight_times_numsample[1,4] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,5,2,1] + [
   x_poison_cat[1,2,1] * ln_numweight_times_numsample[1,5] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,5,2,2] + [
   x_poison_cat[1,2,2] * ln_numweight_times_numsample[1,5] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,5,2,3] + [
   x_poison_cat[1,2,3] * ln_numweight_times_numsample[1,5] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,5,2,4] + [
   x_poison_cat[1,2,4] * ln_numweight_times_numsample[1,5] ] = 0
 pcons_tnc: - tnc_ln_times_catsamples[1,5,2,5] + [
   x_poison_cat[1,2,5] * ln_numweight_times_numsample[1,5] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,1,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,1,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,1,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,1,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,1,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,1,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,1,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,1,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,1,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,1,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,2,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,1,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,2,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,1,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,2,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,1,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,2,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,1,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,2,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,1,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,3,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,1,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,3,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,1,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,3,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,1,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,3,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,1,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,3,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,1,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,4,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,1,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,4,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,1,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,4,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,1,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,4,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,1,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,4,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,1,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,5,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,1,5] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,5,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,1,5] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,5,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,1,5] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,5,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,1,5] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,1,5,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,1,5] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,1,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,2,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,1,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,2,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,1,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,2,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,1,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,2,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,1,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,2,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,2,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,2,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,2,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,2,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,2,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,2,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,2,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,2,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,2,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,2,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,3,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,2,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,3,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,2,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,3,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,2,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,3,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,2,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,3,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,2,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,4,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,2,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,4,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,2,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,4,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,2,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,4,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,2,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,4,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,2,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,5,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,2,5] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,5,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,2,5] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,5,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,2,5] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,5,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,2,5] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,2,5,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,2,5] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,1,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,3,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,1,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,3,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,1,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,3,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,1,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,3,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,1,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,3,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,2,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,3,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,2,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,3,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,2,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,3,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,2,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,3,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,2,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,3,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,3,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,3,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,3,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,3,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,3,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,3,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,3,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,3,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,3,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,3,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,4,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,3,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,4,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,3,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,4,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,3,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,4,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,3,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,3,4,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,3,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,1,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,4,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,1,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,4,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,1,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,4,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,1,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,4,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,1,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,4,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,2,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,4,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,2,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,4,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,2,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,4,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,2,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,4,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,2,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,4,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,3,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,4,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,3,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,4,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,3,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,4,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,3,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,4,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,3,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,4,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,4,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,4,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,4,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,4,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,4,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,4,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,4,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,4,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,4,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,4,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,5,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,4,5] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,5,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,4,5] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,5,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,4,5] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,5,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,4,5] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,5,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,4,5] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,6,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,4,6] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,6,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,4,6] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,6,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,4,6] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,6,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,4,6] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,4,6,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,4,6] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,1,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,5,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,1,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,5,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,1,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,5,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,1,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,5,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,1,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,5,1] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,2,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,5,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,2,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,5,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,2,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,5,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,2,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,5,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,2,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,5,2] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,3,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,5,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,3,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,5,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,3,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,5,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,3,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,5,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,3,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,5,3] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,4,2,1] + [
   x_poison_cat[1,2,1] * lc_catweight_times_catsample[1,5,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,4,2,2] + [
   x_poison_cat[1,2,2] * lc_catweight_times_catsample[1,5,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,4,2,3] + [
   x_poison_cat[1,2,3] * lc_catweight_times_catsample[1,5,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,4,2,4] + [
   x_poison_cat[1,2,4] * lc_catweight_times_catsample[1,5,4] ] = 0
 pcons_tcc: - tcc_lc_times_catsample[1,5,4,2,5] + [
   x_poison_cat[1,2,5] * lc_catweight_times_catsample[1,5,4] ] = 0
 pcons_lc: - lc_catweight_times_catsample[1,2,1] + [
   x_poison_cat[1,2,1] * self.weights_cat[2,1] ] = 0
 pcons_lc: - lc_catweight_times_catsample[1,2,2] + [
   x_poison_cat[1,2,2] * self.weights_cat[2,2] ] = 0
 pcons_lc: - lc_catweight_times_catsample[1,2,3] + [
   x_poison_cat[1,2,3] * self.weights_cat[2,3] ] = 0
 pcons_lc: - lc_catweight_times_catsample[1,2,4] + [
   x_poison_cat[1,2,4] * self.weights_cat[2,4] ] = 0
 pcons_lc: - lc_catweight_times_catsample[1,2,5] + [
   x_poison_cat[1,2,5] * self.weights_cat[2,5] ] = 0
Bounds
 -10 <= weights_num[1] <= 10
 -10 <= weights_num[2] <= 10
 -10 <= weights_num[3] <= 10
 -10 <= weights_num[4] <= 10
 -10 <= weights_num[5] <= 10
 -10 <= self.weights_cat[1,1] <= 10
 -10 <= self.weights_cat[1,2] <= 10
 -10 <= self.weights_cat[1,3] <= 10
 -10 <= self.weights_cat[1,4] <= 10
 -10 <= self.weights_cat[1,5] <= 10
 -10 <= self.weights_cat[2,1] <= 10
 -10 <= self.weights_cat[2,2] <= 10
 -10 <= self.weights_cat[2,3] <= 10
 -10 <= self.weights_cat[2,4] <= 10
 -10 <= self.weights_cat[2,5] <= 10
 -10 <= self.weights_cat[3,1] <= 10
 -10 <= self.weights_cat[3,2] <= 10
 -10 <= self.weights_cat[3,3] <= 10
 -10 <= self.weights_cat[3,4] <= 10
 -10 <= self.weights_cat[4,1] <= 10
 -10 <= self.weights_cat[4,2] <= 10
 -10 <= self.weights_cat[4,3] <= 10
 -10 <= self.weights_cat[4,4] <= 10
 -10 <= self.weights_cat[4,5] <= 10
 -10 <= self.weights_cat[4,6] <= 10
 -10 <= self.weights_cat[5,1] <= 10
 -10 <= self.weights_cat[5,2] <= 10
 -10 <= self.weights_cat[5,3] <= 10
 -10 <= self.weights_cat[5,4] <= 10
 -10 <= bias <= 10
 -10 <= tnn_ln_times_numsamples[1,1,1] <= 10
 -10 <= tnn_ln_times_numsamples[1,1,2] <= 10
 -10 <= tnn_ln_times_numsamples[1,1,3] <= 10
 -10 <= tnn_ln_times_numsamples[1,1,4] <= 10
 -10 <= tnn_ln_times_numsamples[1,1,5] <= 10
 -10 <= tnn_ln_times_numsamples[1,2,1] <= 10
 -10 <= tnn_ln_times_numsamples[1,2,2] <= 10
 -10 <= tnn_ln_times_numsamples[1,2,3] <= 10
 -10 <= tnn_ln_times_numsamples[1,2,4] <= 10
 -10 <= tnn_ln_times_numsamples[1,2,5] <= 10
 -10 <= tnn_ln_times_numsamples[1,3,1] <= 10
 -10 <= tnn_ln_times_numsamples[1,3,2] <= 10
 -10 <= tnn_ln_times_numsamples[1,3,3] <= 10
 -10 <= tnn_ln_times_numsamples[1,3,4] <= 10
 -10 <= tnn_ln_times_numsamples[1,3,5] <= 10
 -10 <= tnn_ln_times_numsamples[1,4,1] <= 10
 -10 <= tnn_ln_times_numsamples[1,4,2] <= 10
 -10 <= tnn_ln_times_numsamples[1,4,3] <= 10
 -10 <= tnn_ln_times_numsamples[1,4,4] <= 10
 -10 <= tnn_ln_times_numsamples[1,4,5] <= 10
 -10 <= tnn_ln_times_numsamples[1,5,1] <= 10
 -10 <= tnn_ln_times_numsamples[1,5,2] <= 10
 -10 <= tnn_ln_times_numsamples[1,5,3] <= 10
 -10 <= tnn_ln_times_numsamples[1,5,4] <= 10
 -10 <= tnn_ln_times_numsamples[1,5,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,1,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,2,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,3,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,4,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,5,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,1,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,2,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,3,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,4,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,5,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,1,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,2,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,3,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,4,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,1,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,2,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,3,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,4,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,5,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,6,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,1,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,2,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,3,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,4,1] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,1,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,2,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,3,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,4,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,5,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,1,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,2,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,3,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,4,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,5,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,1,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,2,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,3,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,4,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,1,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,2,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,3,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,4,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,5,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,6,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,1,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,2,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,3,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,4,2] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,1,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,2,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,3,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,4,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,5,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,1,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,2,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,3,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,4,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,5,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,1,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,2,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,3,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,4,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,1,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,2,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,3,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,4,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,5,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,6,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,1,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,2,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,3,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,4,3] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,1,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,2,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,3,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,4,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,5,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,1,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,2,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,3,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,4,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,5,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,1,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,2,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,3,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,4,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,1,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,2,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,3,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,4,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,5,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,6,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,1,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,2,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,3,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,4,4] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,1,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,2,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,3,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,4,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,1,5,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,1,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,2,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,3,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,4,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,2,5,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,1,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,2,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,3,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,3,4,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,1,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,2,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,3,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,4,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,5,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,4,6,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,1,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,2,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,3,5] <= 10
 -10 <= tcn_lc_times_numsamples[1,5,4,5] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,1,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,1,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,1,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,1,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,1,5] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,2,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,2,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,2,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,2,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,2,5] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,3,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,3,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,3,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,3,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,4,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,4,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,4,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,4,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,4,5] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,4,6] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,5,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,5,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,5,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,1,5,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,1,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,1,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,1,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,1,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,1,5] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,2,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,2,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,2,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,2,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,2,5] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,3,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,3,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,3,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,3,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,4,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,4,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,4,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,4,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,4,5] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,4,6] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,5,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,5,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,5,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,2,5,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,1,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,1,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,1,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,1,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,1,5] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,2,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,2,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,2,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,2,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,2,5] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,3,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,3,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,3,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,3,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,4,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,4,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,4,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,4,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,4,5] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,4,6] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,5,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,5,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,5,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,3,5,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,1,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,1,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,1,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,1,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,1,5] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,2,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,2,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,2,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,2,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,2,5] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,3,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,3,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,3,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,3,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,4,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,4,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,4,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,4,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,4,5] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,4,6] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,5,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,5,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,5,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,4,5,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,1,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,1,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,1,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,1,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,1,5] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,2,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,2,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,2,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,2,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,2,5] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,3,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,3,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,3,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,3,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,4,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,4,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,4,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,4,4] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,4,5] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,4,6] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,5,1] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,5,2] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,5,3] <= 10
 -10 <= tnc_ln_times_catsamples[1,5,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,1,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,2,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,3,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,4,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,1,5,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,1,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,2,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,3,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,4,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,2,5,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,1,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,2,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,3,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,3,4,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,1,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,2,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,3,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,4,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,5,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,4,6,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,1,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,2,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,3,5,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,1,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,1,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,1,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,1,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,1,5] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,2,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,2,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,2,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,2,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,2,5] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,3,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,3,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,3,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,3,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,4,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,4,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,4,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,4,4] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,4,5] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,4,6] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,5,1] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,5,2] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,5,3] <= 10
 -10 <= tcc_lc_times_catsample[1,5,4,5,4] <= 10
 -10 <= ln_numweight_times_numsample[1,1] <= 10
 -10 <= ln_numweight_times_numsample[1,2] <= 10
 -10 <= ln_numweight_times_numsample[1,3] <= 10
 -10 <= ln_numweight_times_numsample[1,4] <= 10
 -10 <= ln_numweight_times_numsample[1,5] <= 10
 -10 <= lc_catweight_times_catsample[1,1,1] <= 10
 -10 <= lc_catweight_times_catsample[1,1,2] <= 10
 -10 <= lc_catweight_times_catsample[1,1,3] <= 10
 -10 <= lc_catweight_times_catsample[1,1,4] <= 10
 -10 <= lc_catweight_times_catsample[1,1,5] <= 10
 -10 <= lc_catweight_times_catsample[1,2,1] <= 10
 -10 <= lc_catweight_times_catsample[1,2,2] <= 10
 -10 <= lc_catweight_times_catsample[1,2,3] <= 10
 -10 <= lc_catweight_times_catsample[1,2,4] <= 10
 -10 <= lc_catweight_times_catsample[1,2,5] <= 10
 -10 <= lc_catweight_times_catsample[1,3,1] <= 10
 -10 <= lc_catweight_times_catsample[1,3,2] <= 10
 -10 <= lc_catweight_times_catsample[1,3,3] <= 10
 -10 <= lc_catweight_times_catsample[1,3,4] <= 10
 -10 <= lc_catweight_times_catsample[1,4,1] <= 10
 -10 <= lc_catweight_times_catsample[1,4,2] <= 10
 -10 <= lc_catweight_times_catsample[1,4,3] <= 10
 -10 <= lc_catweight_times_catsample[1,4,4] <= 10
 -10 <= lc_catweight_times_catsample[1,4,5] <= 10
 -10 <= lc_catweight_times_catsample[1,4,6] <= 10
 -10 <= lc_catweight_times_catsample[1,5,1] <= 10
 -10 <= lc_catweight_times_catsample[1,5,2] <= 10
 -10 <= lc_catweight_times_catsample[1,5,3] <= 10
 -10 <= lc_catweight_times_catsample[1,5,4] <= 10
 Constant = 1
Binaries
 x_poison_cat[1,2,1] x_poison_cat[1,2,2] x_poison_cat[1,2,3]
 x_poison_cat[1,2,4] x_poison_cat[1,2,5]
End
