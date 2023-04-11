\ Model Poisoning_Attack
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 2.09193167615609 weights_num[1] - 2.442915335832986 weights_num[2]
   - 1.875850576308846 self.weights_cat[1,4]
   - 1.875850576308846 self.weights_cat[2,3] - 2.979030690181919 bias
   + 0.4863334165213865 Constant + [ 4.814814814814815 weights_num[1] ^2
   + 11.39259259259259 weights_num[1] * weights_num[2]
   + 8.888888888888889 weights_num[1] * self.weights_cat[1,4]
   + 8.888888888888889 weights_num[1] * self.weights_cat[2,3]
   + 13.77777777777778 weights_num[1] * bias
   + 6.868333333333334 weights_num[2] ^2
   + 10.86666666666667 weights_num[2] * self.weights_cat[1,4]
   + 10.86666666666667 weights_num[2] * self.weights_cat[2,3]
   + 16.33333333333334 weights_num[2] * bias + 6 self.weights_cat[1,4] ^2
   + 12 self.weights_cat[1,4] * self.weights_cat[2,3]
   + 12 self.weights_cat[1,4] * bias + 6 self.weights_cat[2,3] ^2
   + 12 self.weights_cat[2,3] * bias + 10 bias ^2 ] / 2 
Subject To
 cons_first_order_optimality_conditions_num_weights[1]:
   6.137263794406651 weights_num[1] + 5.696296296296296 weights_num[2]
   + 4.444444444444445 self.weights_cat[1,4]
   + 4.444444444444445 self.weights_cat[2,3] + 6.888888888888888 bias
   = 2.091931676156089
 cons_first_order_optimality_conditions_num_weights[2]:
   5.696296296296296 weights_num[1] + 8.190782312925171 weights_num[2]
   + 5.433333333333334 self.weights_cat[1,4]
   + 5.433333333333334 self.weights_cat[2,3] + 8.166666666666666 bias
   = 2.442915335832986
 cons_first_order_optimality_conditions_cat_weights[1,1]:
   1.322448979591837 self.weights_cat[1,1] = 0
 cons_first_order_optimality_conditions_cat_weights[1,2]:
   1.322448979591837 self.weights_cat[1,2] = 0
 cons_first_order_optimality_conditions_cat_weights[1,3]:
   1.322448979591837 self.weights_cat[1,3] = 0
 cons_first_order_optimality_conditions_cat_weights[1,4]:
   4.444444444444445 weights_num[1] + 5.433333333333334 weights_num[2]
   + 7.322448979591837 self.weights_cat[1,4] + 6 self.weights_cat[2,3]
   + 6 bias = 1.875850576308846
 cons_first_order_optimality_conditions_cat_weights[1,5]:
   1.322448979591837 self.weights_cat[1,5] = 0
 cons_first_order_optimality_conditions_cat_weights[2,1]:
   1.322448979591837 self.weights_cat[2,1] = 0
 cons_first_order_optimality_conditions_cat_weights[2,2]:
   1.322448979591837 self.weights_cat[2,2] = 0
 cons_first_order_optimality_conditions_cat_weights[2,3]:
   4.444444444444445 weights_num[1] + 5.433333333333334 weights_num[2]
   + 6 self.weights_cat[1,4] + 7.322448979591837 self.weights_cat[2,3]
   + 6 bias = 1.875850576308846
 cons_first_order_optimality_conditions_cat_weights[2,4]:
   1.322448979591837 self.weights_cat[2,4] = 0
 cons_first_order_optimality_conditions_cat_weights[2,5]:
   1.322448979591837 self.weights_cat[2,5] = 0
 cons_first_order_optimality_conditions_bias:
   6.888888888888889 weights_num[1] + 8.166666666666668 weights_num[2]
   + 6 self.weights_cat[1,4] + 6 self.weights_cat[2,3] + 10 bias
   = 2.979030690181919
Bounds
 -1 <= weights_num[1] <= 1
 -1 <= weights_num[2] <= 1
 -1 <= self.weights_cat[1,1] <= 1
 -1 <= self.weights_cat[1,2] <= 1
 -1 <= self.weights_cat[1,3] <= 1
 -1 <= self.weights_cat[1,4] <= 1
 -1 <= self.weights_cat[1,5] <= 1
 -1 <= self.weights_cat[2,1] <= 1
 -1 <= self.weights_cat[2,2] <= 1
 -1 <= self.weights_cat[2,3] <= 1
 -1 <= self.weights_cat[2,4] <= 1
 -1 <= self.weights_cat[2,5] <= 1
 -1 <= bias <= 1
 Constant = 1
End
