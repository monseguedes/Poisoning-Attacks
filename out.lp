\ Model Poisoning_Attack
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 1.830786363448961 weights_num[1] - 0.5723908051713622 weights_num[2]
   - 6.999058772701321 weights_num[3] - 5.942820441605331 weights_num[4]
   - 8.012721663054043 weights_num[5] - 8.144429014488729 weights_num[6]
   - 1.336355413934704 weights_num[7] - 1.35412651648283 weights_num[8]
   - 0.6470305475257392 weights_num[9] - 3.096660376317596 weights_num[10]
   - 10.30579086238022 bias + 1.513904248937327 Constant + [
   1.334204279222422 weights_num[1] ^2
   + 0.8104909680100096 weights_num[1] * weights_num[2]
   + 8.631049467909163 weights_num[1] * weights_num[3]
   + 7.845033533699085 weights_num[1] * weights_num[4]
   + 10.27384259064108 weights_num[1] * weights_num[5]
   + 10.32308122226911 weights_num[1] * weights_num[6]
   + 1.618874159433467 weights_num[1] * weights_num[7]
   + 1.70097688503149 weights_num[1] * weights_num[8]
   + 1.098613590349416 weights_num[1] * weights_num[9]
   + 4.112101773651749 weights_num[1] * weights_num[10]
   + 13.5904098183021 weights_num[1] * bias
   + 0.1788792070797679 weights_num[2] ^2
   + 2.639644145302141 weights_num[2] * weights_num[3]
   + 2.639278786604032 weights_num[2] * weights_num[4]
   + 2.91417955515453 weights_num[2] * weights_num[5]
   + 3.204359687458612 weights_num[2] * weights_num[6]
   + 0.4443955620497193 weights_num[2] * weights_num[7]
   + 0.5088674089491999 weights_num[2] * weights_num[8]
   + 0.2622583725338579 weights_num[2] * weights_num[9]
   + 1.372347689218551 weights_num[2] * weights_num[10]
   + 4.266704059454528 weights_num[2] * bias
   + 16.93827160493828 weights_num[3] ^2
   + 28.94444444444445 weights_num[3] * weights_num[4]
   + 40.45410628019323 weights_num[3] * weights_num[5]
   + 40.4 weights_num[3] * weights_num[6]
   + 6.665304369452095 weights_num[3] * weights_num[7]
   + 6.474096547349752 weights_num[3] * weights_num[8]
   + 3.189983251034161 weights_num[3] * weights_num[9]
   + 15.08770928462709 weights_num[3] * weights_num[10]
   + 50.66666666666668 weights_num[3] * bias + 14.1875 weights_num[4] ^2
   + 34.39492753623188 weights_num[4] * weights_num[5]
   + 36.66666666666667 weights_num[4] * weights_num[6]
   + 5.700539127066113 weights_num[4] * weights_num[7]
   + 5.527873217284933 weights_num[4] * weights_num[8]
   + 3.18217226445605 weights_num[4] * weights_num[9]
   + 12.62970890410959 weights_num[4] * weights_num[10]
   + 46 weights_num[4] * bias + 25.87114051669818 weights_num[5] ^2
   + 50.10338164251208 weights_num[5] * weights_num[6]
   + 8.080702728869724 weights_num[5] * weights_num[7]
   + 7.883588989317462 weights_num[5] * weights_num[8]
   + 3.90375950665213 weights_num[5] * weights_num[9]
   + 18.5911628945801 weights_num[5] * weights_num[10]
   + 61.76811594202898 weights_num[5] * bias
   + 26.44833333333333 weights_num[6] ^2
   + 8.03542151629936 weights_num[6] * weights_num[7]
   + 7.666610616519259 weights_num[6] * weights_num[8]
   + 4.39607157239952 weights_num[6] * weights_num[9]
   + 17.90256849315069 weights_num[6] * weights_num[10]
   + 61.53333333333333 weights_num[6] * bias
   + 0.9653586468465886 weights_num[7] ^2
   + 1.218133799298307 weights_num[7] * weights_num[8]
   + 0.6263915556940579 weights_num[7] * weights_num[9]
   + 2.998683686324247 weights_num[7] * weights_num[10]
   + 9.833705234159778 weights_num[7] * bias
   + 0.7309381712696021 weights_num[8] ^2
   + 0.5785349464277185 weights_num[8] * weights_num[9]
   + 2.756573484393791 weights_num[8] * weights_num[10]
   + 9.688049862625359 weights_num[8] * bias
   + 0.5873936999352183 weights_num[9] ^2
   + 1.652107315569069 weights_num[9] * weights_num[10]
   + 5.2235757699671 weights_num[9] * bias
   + 4.437594925759992 weights_num[10] ^2
   + 23.26198630136987 weights_num[10] * bias + 40 bias ^2 ] / 2 
Subject To
 cons_first_order_optimality_conditions_bias:
   6.795204909151048 weights_num[1] + 2.133352029727265 weights_num[2]
   + 25.33333333333334 weights_num[3] + 23 weights_num[4]
   + 30.88405797101449 weights_num[5] + 30.76666666666667 weights_num[6]
   + 4.916852617079889 weights_num[7] + 4.844024931312678 weights_num[8]
   + 2.61178788498355 weights_num[9] + 11.63099315068493 weights_num[10]
   + 42 bias + 2 ln_numweight_times_numsample[1,1]
   + 2 ln_numweight_times_numsample[1,2]
   + 2 ln_numweight_times_numsample[1,3]
   + 2 ln_numweight_times_numsample[1,4]
   + 2 ln_numweight_times_numsample[1,5]
   + 2 ln_numweight_times_numsample[1,6]
   + 2 ln_numweight_times_numsample[1,7]
   + 2 ln_numweight_times_numsample[1,8]
   + 2 ln_numweight_times_numsample[1,9]
   + 2 ln_numweight_times_numsample[1,10] = 10.53659213998056
 cons_first_order_optimality_conditions_pnum_weights[1]:
   - 0.2308012776003332 x_poison_num[1,1]
   + 2.656653258814259 weights_num[1] + 0.4052454840050049 weights_num[2]
   + 4.315524733954581 weights_num[3] + 3.922516766849542 weights_num[4]
   + 5.136921295320539 weights_num[5] + 5.161540611134557 weights_num[6]
   + 0.8094370797167334 weights_num[7] + 0.8504884425157446 weights_num[8]
   + 0.5493067951747078 weights_num[9] + 2.056050886825874 weights_num[10]
   + 6.795204909151049 bias + 2 tnn_ln_times_numsamples[1,1,1]
   + 2 tnn_ln_times_numsamples[1,2,1] + 2 tnn_ln_times_numsamples[1,3,1]
   + 2 tnn_ln_times_numsamples[1,4,1] + 2 tnn_ln_times_numsamples[1,5,1]
   + 2 tnn_ln_times_numsamples[1,6,1] + 2 tnn_ln_times_numsamples[1,7,1]
   + 2 tnn_ln_times_numsamples[1,8,1] + 2 tnn_ln_times_numsamples[1,9,1]
   + 2 tnn_ln_times_numsamples[1,10,1] + [ 2 x_poison_num[1,1] * bias ]
   = 1.830786363448961
 cons_first_order_optimality_conditions_pnum_weights[2]:
   - 0.2308012776003332 x_poison_num[1,2]
   + 0.4052454840050049 weights_num[1] + 1.501328186671604 weights_num[2]
   + 1.31982207265107 weights_num[3] + 1.319639393302016 weights_num[4]
   + 1.457089777577265 weights_num[5] + 1.602179843729306 weights_num[6]
   + 0.2221977810248597 weights_num[7] + 0.2544337044746 weights_num[8]
   + 0.131129186266929 weights_num[9] + 0.6861738446092752 weights_num[10]
   + 2.133352029727265 bias + 2 tnn_ln_times_numsamples[1,1,2]
   + 2 tnn_ln_times_numsamples[1,2,2] + 2 tnn_ln_times_numsamples[1,3,2]
   + 2 tnn_ln_times_numsamples[1,4,2] + 2 tnn_ln_times_numsamples[1,5,2]
   + 2 tnn_ln_times_numsamples[1,6,2] + 2 tnn_ln_times_numsamples[1,7,2]
   + 2 tnn_ln_times_numsamples[1,8,2] + 2 tnn_ln_times_numsamples[1,9,2]
   + 2 tnn_ln_times_numsamples[1,10,2] + [ 2 x_poison_num[1,2] * bias ]
   = 0.5723908051713622
 cons_first_order_optimality_conditions_pnum_weights[3]:
   - 0.2308012776003332 x_poison_num[1,3]
   + 4.315524733954581 weights_num[1] + 1.31982207265107 weights_num[2]
   + 18.26072058453011 weights_num[3] + 14.47222222222222 weights_num[4]
   + 20.22705314009662 weights_num[5] + 20.2 weights_num[6]
   + 3.332652184726048 weights_num[7] + 3.237048273674877 weights_num[8]
   + 1.594991625517081 weights_num[9] + 7.543854642313547 weights_num[10]
   + 25.33333333333333 bias + 2 tnn_ln_times_numsamples[1,1,3]
   + 2 tnn_ln_times_numsamples[1,2,3] + 2 tnn_ln_times_numsamples[1,3,3]
   + 2 tnn_ln_times_numsamples[1,4,3] + 2 tnn_ln_times_numsamples[1,5,3]
   + 2 tnn_ln_times_numsamples[1,6,3] + 2 tnn_ln_times_numsamples[1,7,3]
   + 2 tnn_ln_times_numsamples[1,8,3] + 2 tnn_ln_times_numsamples[1,9,3]
   + 2 tnn_ln_times_numsamples[1,10,3] + [ 2 x_poison_num[1,3] * bias ]
   = 6.999058772701322
 cons_first_order_optimality_conditions_pnum_weights[4]:
   - 0.2308012776003332 x_poison_num[1,4]
   + 3.922516766849543 weights_num[1] + 1.319639393302016 weights_num[2]
   + 14.47222222222222 weights_num[3] + 15.50994897959184 weights_num[4]
   + 17.19746376811595 weights_num[5] + 18.33333333333334 weights_num[6]
   + 2.850269563533057 weights_num[7] + 2.763936608642466 weights_num[8]
   + 1.591086132228025 weights_num[9] + 6.314854452054794 weights_num[10]
   + 23 bias + 2 tnn_ln_times_numsamples[1,1,4]
   + 2 tnn_ln_times_numsamples[1,2,4] + 2 tnn_ln_times_numsamples[1,3,4]
   + 2 tnn_ln_times_numsamples[1,4,4] + 2 tnn_ln_times_numsamples[1,5,4]
   + 2 tnn_ln_times_numsamples[1,6,4] + 2 tnn_ln_times_numsamples[1,7,4]
   + 2 tnn_ln_times_numsamples[1,8,4] + 2 tnn_ln_times_numsamples[1,9,4]
   + 2 tnn_ln_times_numsamples[1,10,4] + [ 2 x_poison_num[1,4] * bias ]
   = 5.942820441605332
 cons_first_order_optimality_conditions_pnum_weights[5]:
   - 0.2308012776003332 x_poison_num[1,5] + 5.13692129532054 weights_num[1]
   + 1.457089777577265 weights_num[2] + 20.22705314009662 weights_num[3]
   + 17.19746376811595 weights_num[4] + 27.19358949629001 weights_num[5]
   + 25.05169082125604 weights_num[6] + 4.040351364434862 weights_num[7]
   + 3.941794494658731 weights_num[8] + 1.951879753326065 weights_num[9]
   + 9.295581447290052 weights_num[10] + 30.88405797101449 bias
   + 2 tnn_ln_times_numsamples[1,1,5] + 2 tnn_ln_times_numsamples[1,2,5]
   + 2 tnn_ln_times_numsamples[1,3,5] + 2 tnn_ln_times_numsamples[1,4,5]
   + 2 tnn_ln_times_numsamples[1,5,5] + 2 tnn_ln_times_numsamples[1,6,5]
   + 2 tnn_ln_times_numsamples[1,7,5] + 2 tnn_ln_times_numsamples[1,8,5]
   + 2 tnn_ln_times_numsamples[1,9,5] + 2 tnn_ln_times_numsamples[1,10,5]
   + [ 2 x_poison_num[1,5] * bias ] = 8.012721663054043
 cons_first_order_optimality_conditions_pnum_weights[6]:
   - 0.2308012776003332 x_poison_num[1,6]
   + 5.161540611134556 weights_num[1] + 1.602179843729306 weights_num[2]
   + 20.2 weights_num[3] + 18.33333333333334 weights_num[4]
   + 25.05169082125604 weights_num[5] + 27.77078231292517 weights_num[6]
   + 4.017710758149677 weights_num[7] + 3.833305308259629 weights_num[8]
   + 2.19803578619976 weights_num[9] + 8.951284246575341 weights_num[10]
   + 30.76666666666667 bias + 2 tnn_ln_times_numsamples[1,1,6]
   + 2 tnn_ln_times_numsamples[1,2,6] + 2 tnn_ln_times_numsamples[1,3,6]
   + 2 tnn_ln_times_numsamples[1,4,6] + 2 tnn_ln_times_numsamples[1,5,6]
   + 2 tnn_ln_times_numsamples[1,6,6] + 2 tnn_ln_times_numsamples[1,7,6]
   + 2 tnn_ln_times_numsamples[1,8,6] + 2 tnn_ln_times_numsamples[1,9,6]
   + 2 tnn_ln_times_numsamples[1,10,6] + [ 2 x_poison_num[1,6] * bias ]
   = 8.144429014488729
 cons_first_order_optimality_conditions_pnum_weights[7]:
   - 0.2308012776003332 x_poison_num[1,7]
   + 0.8094370797167333 weights_num[1] + 0.2221977810248596 weights_num[2]
   + 3.332652184726047 weights_num[3] + 2.850269563533058 weights_num[4]
   + 4.040351364434861 weights_num[5] + 4.017710758149679 weights_num[6]
   + 2.287807626438425 weights_num[7] + 0.6090668996491538 weights_num[8]
   + 0.3131957778470288 weights_num[9] + 1.499341843162124 weights_num[10]
   + 4.91685261707989 bias + 2 tnn_ln_times_numsamples[1,1,7]
   + 2 tnn_ln_times_numsamples[1,2,7] + 2 tnn_ln_times_numsamples[1,3,7]
   + 2 tnn_ln_times_numsamples[1,4,7] + 2 tnn_ln_times_numsamples[1,5,7]
   + 2 tnn_ln_times_numsamples[1,6,7] + 2 tnn_ln_times_numsamples[1,7,7]
   + 2 tnn_ln_times_numsamples[1,8,7] + 2 tnn_ln_times_numsamples[1,9,7]
   + 2 tnn_ln_times_numsamples[1,10,7] + [ 2 x_poison_num[1,7] * bias ]
   = 1.336355413934703
 cons_first_order_optimality_conditions_pnum_weights[8]:
   - 0.2308012776003332 x_poison_num[1,8]
   + 0.8504884425157446 weights_num[1] + 0.2544337044746 weights_num[2]
   + 3.237048273674876 weights_num[3] + 2.763936608642466 weights_num[4]
   + 3.941794494658732 weights_num[5] + 3.83330530825963 weights_num[6]
   + 0.6090668996491538 weights_num[7] + 2.053387150861439 weights_num[8]
   + 0.2892674732138591 weights_num[9] + 1.378286742196895 weights_num[10]
   + 4.844024931312679 bias + 2 tnn_ln_times_numsamples[1,1,8]
   + 2 tnn_ln_times_numsamples[1,2,8] + 2 tnn_ln_times_numsamples[1,3,8]
   + 2 tnn_ln_times_numsamples[1,4,8] + 2 tnn_ln_times_numsamples[1,5,8]
   + 2 tnn_ln_times_numsamples[1,6,8] + 2 tnn_ln_times_numsamples[1,7,8]
   + 2 tnn_ln_times_numsamples[1,8,8] + 2 tnn_ln_times_numsamples[1,9,8]
   + 2 tnn_ln_times_numsamples[1,10,8] + [ 2 x_poison_num[1,8] * bias ]
   = 1.35412651648283
 cons_first_order_optimality_conditions_pnum_weights[9]:
   - 0.2308012776003332 x_poison_num[1,9]
   + 0.5493067951747078 weights_num[1] + 0.131129186266929 weights_num[2]
   + 1.59499162551708 weights_num[3] + 1.591086132228025 weights_num[4]
   + 1.951879753326065 weights_num[5] + 2.198035786199761 weights_num[6]
   + 0.3131957778470288 weights_num[7] + 0.2892674732138591 weights_num[8]
   + 1.909842679527054 weights_num[9] + 0.8260536577845347 weights_num[10]
   + 2.61178788498355 bias + 2 tnn_ln_times_numsamples[1,1,9]
   + 2 tnn_ln_times_numsamples[1,2,9] + 2 tnn_ln_times_numsamples[1,3,9]
   + 2 tnn_ln_times_numsamples[1,4,9] + 2 tnn_ln_times_numsamples[1,5,9]
   + 2 tnn_ln_times_numsamples[1,6,9] + 2 tnn_ln_times_numsamples[1,7,9]
   + 2 tnn_ln_times_numsamples[1,8,9] + 2 tnn_ln_times_numsamples[1,9,9]
   + 2 tnn_ln_times_numsamples[1,10,9] + [ 2 x_poison_num[1,9] * bias ]
   = 0.6470305475257391
 cons_first_order_optimality_conditions_pnum_weights[10]:
   - 0.2308012776003332 x_poison_num[1,10]
   + 2.056050886825874 weights_num[1] + 0.6861738446092754 weights_num[2]
   + 7.543854642313547 weights_num[3] + 6.314854452054794 weights_num[4]
   + 9.295581447290054 weights_num[5] + 8.951284246575343 weights_num[6]
   + 1.499341843162124 weights_num[7] + 1.378286742196895 weights_num[8]
   + 0.8260536577845347 weights_num[9] + 5.76004390535183 weights_num[10]
   + 11.63099315068493 bias + 2 tnn_ln_times_numsamples[1,1,10]
   + 2 tnn_ln_times_numsamples[1,2,10] + 2 tnn_ln_times_numsamples[1,3,10]
   + 2 tnn_ln_times_numsamples[1,4,10] + 2 tnn_ln_times_numsamples[1,5,10]
   + 2 tnn_ln_times_numsamples[1,6,10] + 2 tnn_ln_times_numsamples[1,7,10]
   + 2 tnn_ln_times_numsamples[1,8,10] + 2 tnn_ln_times_numsamples[1,9,10]
   + 2 tnn_ln_times_numsamples[1,10,10] + [ 2 x_poison_num[1,10] * bias ]
   = 3.096660376317597
 pons_tnn[1,1,1]: - tnn_ln_times_numsamples[1,1,1] + [
   x_poison_num[1,1] * ln_numweight_times_numsample[1,1] ] = 0
 pons_tnn[1,1,2]: - tnn_ln_times_numsamples[1,1,2] + [
   x_poison_num[1,2] * ln_numweight_times_numsample[1,1] ] = 0
 pons_tnn[1,1,3]: - tnn_ln_times_numsamples[1,1,3] + [
   x_poison_num[1,3] * ln_numweight_times_numsample[1,1] ] = 0
 pons_tnn[1,1,4]: - tnn_ln_times_numsamples[1,1,4] + [
   x_poison_num[1,4] * ln_numweight_times_numsample[1,1] ] = 0
 pons_tnn[1,1,5]: - tnn_ln_times_numsamples[1,1,5] + [
   x_poison_num[1,5] * ln_numweight_times_numsample[1,1] ] = 0
 pons_tnn[1,1,6]: - tnn_ln_times_numsamples[1,1,6] + [
   x_poison_num[1,6] * ln_numweight_times_numsample[1,1] ] = 0
 pons_tnn[1,1,7]: - tnn_ln_times_numsamples[1,1,7] + [
   x_poison_num[1,7] * ln_numweight_times_numsample[1,1] ] = 0
 pons_tnn[1,1,8]: - tnn_ln_times_numsamples[1,1,8] + [
   x_poison_num[1,8] * ln_numweight_times_numsample[1,1] ] = 0
 pons_tnn[1,1,9]: - tnn_ln_times_numsamples[1,1,9] + [
   x_poison_num[1,9] * ln_numweight_times_numsample[1,1] ] = 0
 pons_tnn[1,1,10]: - tnn_ln_times_numsamples[1,1,10] + [
   x_poison_num[1,10] * ln_numweight_times_numsample[1,1] ] = 0
 pons_tnn[1,2,1]: - tnn_ln_times_numsamples[1,2,1] + [
   x_poison_num[1,1] * ln_numweight_times_numsample[1,2] ] = 0
 pons_tnn[1,2,2]: - tnn_ln_times_numsamples[1,2,2] + [
   x_poison_num[1,2] * ln_numweight_times_numsample[1,2] ] = 0
 pons_tnn[1,2,3]: - tnn_ln_times_numsamples[1,2,3] + [
   x_poison_num[1,3] * ln_numweight_times_numsample[1,2] ] = 0
 pons_tnn[1,2,4]: - tnn_ln_times_numsamples[1,2,4] + [
   x_poison_num[1,4] * ln_numweight_times_numsample[1,2] ] = 0
 pons_tnn[1,2,5]: - tnn_ln_times_numsamples[1,2,5] + [
   x_poison_num[1,5] * ln_numweight_times_numsample[1,2] ] = 0
 pons_tnn[1,2,6]: - tnn_ln_times_numsamples[1,2,6] + [
   x_poison_num[1,6] * ln_numweight_times_numsample[1,2] ] = 0
 pons_tnn[1,2,7]: - tnn_ln_times_numsamples[1,2,7] + [
   x_poison_num[1,7] * ln_numweight_times_numsample[1,2] ] = 0
 pons_tnn[1,2,8]: - tnn_ln_times_numsamples[1,2,8] + [
   x_poison_num[1,8] * ln_numweight_times_numsample[1,2] ] = 0
 pons_tnn[1,2,9]: - tnn_ln_times_numsamples[1,2,9] + [
   x_poison_num[1,9] * ln_numweight_times_numsample[1,2] ] = 0
 pons_tnn[1,2,10]: - tnn_ln_times_numsamples[1,2,10] + [
   x_poison_num[1,10] * ln_numweight_times_numsample[1,2] ] = 0
 pons_tnn[1,3,1]: - tnn_ln_times_numsamples[1,3,1] + [
   x_poison_num[1,1] * ln_numweight_times_numsample[1,3] ] = 0
 pons_tnn[1,3,2]: - tnn_ln_times_numsamples[1,3,2] + [
   x_poison_num[1,2] * ln_numweight_times_numsample[1,3] ] = 0
 pons_tnn[1,3,3]: - tnn_ln_times_numsamples[1,3,3] + [
   x_poison_num[1,3] * ln_numweight_times_numsample[1,3] ] = 0
 pons_tnn[1,3,4]: - tnn_ln_times_numsamples[1,3,4] + [
   x_poison_num[1,4] * ln_numweight_times_numsample[1,3] ] = 0
 pons_tnn[1,3,5]: - tnn_ln_times_numsamples[1,3,5] + [
   x_poison_num[1,5] * ln_numweight_times_numsample[1,3] ] = 0
 pons_tnn[1,3,6]: - tnn_ln_times_numsamples[1,3,6] + [
   x_poison_num[1,6] * ln_numweight_times_numsample[1,3] ] = 0
 pons_tnn[1,3,7]: - tnn_ln_times_numsamples[1,3,7] + [
   x_poison_num[1,7] * ln_numweight_times_numsample[1,3] ] = 0
 pons_tnn[1,3,8]: - tnn_ln_times_numsamples[1,3,8] + [
   x_poison_num[1,8] * ln_numweight_times_numsample[1,3] ] = 0
 pons_tnn[1,3,9]: - tnn_ln_times_numsamples[1,3,9] + [
   x_poison_num[1,9] * ln_numweight_times_numsample[1,3] ] = 0
 pons_tnn[1,3,10]: - tnn_ln_times_numsamples[1,3,10] + [
   x_poison_num[1,10] * ln_numweight_times_numsample[1,3] ] = 0
 pons_tnn[1,4,1]: - tnn_ln_times_numsamples[1,4,1] + [
   x_poison_num[1,1] * ln_numweight_times_numsample[1,4] ] = 0
 pons_tnn[1,4,2]: - tnn_ln_times_numsamples[1,4,2] + [
   x_poison_num[1,2] * ln_numweight_times_numsample[1,4] ] = 0
 pons_tnn[1,4,3]: - tnn_ln_times_numsamples[1,4,3] + [
   x_poison_num[1,3] * ln_numweight_times_numsample[1,4] ] = 0
 pons_tnn[1,4,4]: - tnn_ln_times_numsamples[1,4,4] + [
   x_poison_num[1,4] * ln_numweight_times_numsample[1,4] ] = 0
 pons_tnn[1,4,5]: - tnn_ln_times_numsamples[1,4,5] + [
   x_poison_num[1,5] * ln_numweight_times_numsample[1,4] ] = 0
 pons_tnn[1,4,6]: - tnn_ln_times_numsamples[1,4,6] + [
   x_poison_num[1,6] * ln_numweight_times_numsample[1,4] ] = 0
 pons_tnn[1,4,7]: - tnn_ln_times_numsamples[1,4,7] + [
   x_poison_num[1,7] * ln_numweight_times_numsample[1,4] ] = 0
 pons_tnn[1,4,8]: - tnn_ln_times_numsamples[1,4,8] + [
   x_poison_num[1,8] * ln_numweight_times_numsample[1,4] ] = 0
 pons_tnn[1,4,9]: - tnn_ln_times_numsamples[1,4,9] + [
   x_poison_num[1,9] * ln_numweight_times_numsample[1,4] ] = 0
 pons_tnn[1,4,10]: - tnn_ln_times_numsamples[1,4,10] + [
   x_poison_num[1,10] * ln_numweight_times_numsample[1,4] ] = 0
 pons_tnn[1,5,1]: - tnn_ln_times_numsamples[1,5,1] + [
   x_poison_num[1,1] * ln_numweight_times_numsample[1,5] ] = 0
 pons_tnn[1,5,2]: - tnn_ln_times_numsamples[1,5,2] + [
   x_poison_num[1,2] * ln_numweight_times_numsample[1,5] ] = 0
 pons_tnn[1,5,3]: - tnn_ln_times_numsamples[1,5,3] + [
   x_poison_num[1,3] * ln_numweight_times_numsample[1,5] ] = 0
 pons_tnn[1,5,4]: - tnn_ln_times_numsamples[1,5,4] + [
   x_poison_num[1,4] * ln_numweight_times_numsample[1,5] ] = 0
 pons_tnn[1,5,5]: - tnn_ln_times_numsamples[1,5,5] + [
   x_poison_num[1,5] * ln_numweight_times_numsample[1,5] ] = 0
 pons_tnn[1,5,6]: - tnn_ln_times_numsamples[1,5,6] + [
   x_poison_num[1,6] * ln_numweight_times_numsample[1,5] ] = 0
 pons_tnn[1,5,7]: - tnn_ln_times_numsamples[1,5,7] + [
   x_poison_num[1,7] * ln_numweight_times_numsample[1,5] ] = 0
 pons_tnn[1,5,8]: - tnn_ln_times_numsamples[1,5,8] + [
   x_poison_num[1,8] * ln_numweight_times_numsample[1,5] ] = 0
 pons_tnn[1,5,9]: - tnn_ln_times_numsamples[1,5,9] + [
   x_poison_num[1,9] * ln_numweight_times_numsample[1,5] ] = 0
 pons_tnn[1,5,10]: - tnn_ln_times_numsamples[1,5,10] + [
   x_poison_num[1,10] * ln_numweight_times_numsample[1,5] ] = 0
 pons_tnn[1,6,1]: - tnn_ln_times_numsamples[1,6,1] + [
   x_poison_num[1,1] * ln_numweight_times_numsample[1,6] ] = 0
 pons_tnn[1,6,2]: - tnn_ln_times_numsamples[1,6,2] + [
   x_poison_num[1,2] * ln_numweight_times_numsample[1,6] ] = 0
 pons_tnn[1,6,3]: - tnn_ln_times_numsamples[1,6,3] + [
   x_poison_num[1,3] * ln_numweight_times_numsample[1,6] ] = 0
 pons_tnn[1,6,4]: - tnn_ln_times_numsamples[1,6,4] + [
   x_poison_num[1,4] * ln_numweight_times_numsample[1,6] ] = 0
 pons_tnn[1,6,5]: - tnn_ln_times_numsamples[1,6,5] + [
   x_poison_num[1,5] * ln_numweight_times_numsample[1,6] ] = 0
 pons_tnn[1,6,6]: - tnn_ln_times_numsamples[1,6,6] + [
   x_poison_num[1,6] * ln_numweight_times_numsample[1,6] ] = 0
 pons_tnn[1,6,7]: - tnn_ln_times_numsamples[1,6,7] + [
   x_poison_num[1,7] * ln_numweight_times_numsample[1,6] ] = 0
 pons_tnn[1,6,8]: - tnn_ln_times_numsamples[1,6,8] + [
   x_poison_num[1,8] * ln_numweight_times_numsample[1,6] ] = 0
 pons_tnn[1,6,9]: - tnn_ln_times_numsamples[1,6,9] + [
   x_poison_num[1,9] * ln_numweight_times_numsample[1,6] ] = 0
 pons_tnn[1,6,10]: - tnn_ln_times_numsamples[1,6,10] + [
   x_poison_num[1,10] * ln_numweight_times_numsample[1,6] ] = 0
 pons_tnn[1,7,1]: - tnn_ln_times_numsamples[1,7,1] + [
   x_poison_num[1,1] * ln_numweight_times_numsample[1,7] ] = 0
 pons_tnn[1,7,2]: - tnn_ln_times_numsamples[1,7,2] + [
   x_poison_num[1,2] * ln_numweight_times_numsample[1,7] ] = 0
 pons_tnn[1,7,3]: - tnn_ln_times_numsamples[1,7,3] + [
   x_poison_num[1,3] * ln_numweight_times_numsample[1,7] ] = 0
 pons_tnn[1,7,4]: - tnn_ln_times_numsamples[1,7,4] + [
   x_poison_num[1,4] * ln_numweight_times_numsample[1,7] ] = 0
 pons_tnn[1,7,5]: - tnn_ln_times_numsamples[1,7,5] + [
   x_poison_num[1,5] * ln_numweight_times_numsample[1,7] ] = 0
 pons_tnn[1,7,6]: - tnn_ln_times_numsamples[1,7,6] + [
   x_poison_num[1,6] * ln_numweight_times_numsample[1,7] ] = 0
 pons_tnn[1,7,7]: - tnn_ln_times_numsamples[1,7,7] + [
   x_poison_num[1,7] * ln_numweight_times_numsample[1,7] ] = 0
 pons_tnn[1,7,8]: - tnn_ln_times_numsamples[1,7,8] + [
   x_poison_num[1,8] * ln_numweight_times_numsample[1,7] ] = 0
 pons_tnn[1,7,9]: - tnn_ln_times_numsamples[1,7,9] + [
   x_poison_num[1,9] * ln_numweight_times_numsample[1,7] ] = 0
 pons_tnn[1,7,10]: - tnn_ln_times_numsamples[1,7,10] + [
   x_poison_num[1,10] * ln_numweight_times_numsample[1,7] ] = 0
 pons_tnn[1,8,1]: - tnn_ln_times_numsamples[1,8,1] + [
   x_poison_num[1,1] * ln_numweight_times_numsample[1,8] ] = 0
 pons_tnn[1,8,2]: - tnn_ln_times_numsamples[1,8,2] + [
   x_poison_num[1,2] * ln_numweight_times_numsample[1,8] ] = 0
 pons_tnn[1,8,3]: - tnn_ln_times_numsamples[1,8,3] + [
   x_poison_num[1,3] * ln_numweight_times_numsample[1,8] ] = 0
 pons_tnn[1,8,4]: - tnn_ln_times_numsamples[1,8,4] + [
   x_poison_num[1,4] * ln_numweight_times_numsample[1,8] ] = 0
 pons_tnn[1,8,5]: - tnn_ln_times_numsamples[1,8,5] + [
   x_poison_num[1,5] * ln_numweight_times_numsample[1,8] ] = 0
 pons_tnn[1,8,6]: - tnn_ln_times_numsamples[1,8,6] + [
   x_poison_num[1,6] * ln_numweight_times_numsample[1,8] ] = 0
 pons_tnn[1,8,7]: - tnn_ln_times_numsamples[1,8,7] + [
   x_poison_num[1,7] * ln_numweight_times_numsample[1,8] ] = 0
 pons_tnn[1,8,8]: - tnn_ln_times_numsamples[1,8,8] + [
   x_poison_num[1,8] * ln_numweight_times_numsample[1,8] ] = 0
 pons_tnn[1,8,9]: - tnn_ln_times_numsamples[1,8,9] + [
   x_poison_num[1,9] * ln_numweight_times_numsample[1,8] ] = 0
 pons_tnn[1,8,10]: - tnn_ln_times_numsamples[1,8,10] + [
   x_poison_num[1,10] * ln_numweight_times_numsample[1,8] ] = 0
 pons_tnn[1,9,1]: - tnn_ln_times_numsamples[1,9,1] + [
   x_poison_num[1,1] * ln_numweight_times_numsample[1,9] ] = 0
 pons_tnn[1,9,2]: - tnn_ln_times_numsamples[1,9,2] + [
   x_poison_num[1,2] * ln_numweight_times_numsample[1,9] ] = 0
 pons_tnn[1,9,3]: - tnn_ln_times_numsamples[1,9,3] + [
   x_poison_num[1,3] * ln_numweight_times_numsample[1,9] ] = 0
 pons_tnn[1,9,4]: - tnn_ln_times_numsamples[1,9,4] + [
   x_poison_num[1,4] * ln_numweight_times_numsample[1,9] ] = 0
 pons_tnn[1,9,5]: - tnn_ln_times_numsamples[1,9,5] + [
   x_poison_num[1,5] * ln_numweight_times_numsample[1,9] ] = 0
 pons_tnn[1,9,6]: - tnn_ln_times_numsamples[1,9,6] + [
   x_poison_num[1,6] * ln_numweight_times_numsample[1,9] ] = 0
 pons_tnn[1,9,7]: - tnn_ln_times_numsamples[1,9,7] + [
   x_poison_num[1,7] * ln_numweight_times_numsample[1,9] ] = 0
 pons_tnn[1,9,8]: - tnn_ln_times_numsamples[1,9,8] + [
   x_poison_num[1,8] * ln_numweight_times_numsample[1,9] ] = 0
 pons_tnn[1,9,9]: - tnn_ln_times_numsamples[1,9,9] + [
   x_poison_num[1,9] * ln_numweight_times_numsample[1,9] ] = 0
 pons_tnn[1,9,10]: - tnn_ln_times_numsamples[1,9,10] + [
   x_poison_num[1,10] * ln_numweight_times_numsample[1,9] ] = 0
 pons_tnn[1,10,1]: - tnn_ln_times_numsamples[1,10,1] + [
   x_poison_num[1,1] * ln_numweight_times_numsample[1,10] ] = 0
 pons_tnn[1,10,2]: - tnn_ln_times_numsamples[1,10,2] + [
   x_poison_num[1,2] * ln_numweight_times_numsample[1,10] ] = 0
 pons_tnn[1,10,3]: - tnn_ln_times_numsamples[1,10,3] + [
   x_poison_num[1,3] * ln_numweight_times_numsample[1,10] ] = 0
 pons_tnn[1,10,4]: - tnn_ln_times_numsamples[1,10,4] + [
   x_poison_num[1,4] * ln_numweight_times_numsample[1,10] ] = 0
 pons_tnn[1,10,5]: - tnn_ln_times_numsamples[1,10,5] + [
   x_poison_num[1,5] * ln_numweight_times_numsample[1,10] ] = 0
 pons_tnn[1,10,6]: - tnn_ln_times_numsamples[1,10,6] + [
   x_poison_num[1,6] * ln_numweight_times_numsample[1,10] ] = 0
 pons_tnn[1,10,7]: - tnn_ln_times_numsamples[1,10,7] + [
   x_poison_num[1,7] * ln_numweight_times_numsample[1,10] ] = 0
 pons_tnn[1,10,8]: - tnn_ln_times_numsamples[1,10,8] + [
   x_poison_num[1,8] * ln_numweight_times_numsample[1,10] ] = 0
 pons_tnn[1,10,9]: - tnn_ln_times_numsamples[1,10,9] + [
   x_poison_num[1,9] * ln_numweight_times_numsample[1,10] ] = 0
 pons_tnn[1,10,10]: - tnn_ln_times_numsamples[1,10,10] + [
   x_poison_num[1,10] * ln_numweight_times_numsample[1,10] ] = 0
 pcons_ln: - ln_numweight_times_numsample[1,1] + [
   x_poison_num[1,1] * weights_num[1] ] = 0
 pcons_ln: - ln_numweight_times_numsample[1,2] + [
   x_poison_num[1,2] * weights_num[2] ] = 0
 pcons_ln: - ln_numweight_times_numsample[1,3] + [
   x_poison_num[1,3] * weights_num[3] ] = 0
 pcons_ln: - ln_numweight_times_numsample[1,4] + [
   x_poison_num[1,4] * weights_num[4] ] = 0
 pcons_ln: - ln_numweight_times_numsample[1,5] + [
   x_poison_num[1,5] * weights_num[5] ] = 0
 pcons_ln: - ln_numweight_times_numsample[1,6] + [
   x_poison_num[1,6] * weights_num[6] ] = 0
 pcons_ln: - ln_numweight_times_numsample[1,7] + [
   x_poison_num[1,7] * weights_num[7] ] = 0
 pcons_ln: - ln_numweight_times_numsample[1,8] + [
   x_poison_num[1,8] * weights_num[8] ] = 0
 pcons_ln: - ln_numweight_times_numsample[1,9] + [
   x_poison_num[1,9] * weights_num[9] ] = 0
 pcons_ln: - ln_numweight_times_numsample[1,10] + [
   x_poison_num[1,10] * weights_num[10] ] = 0
Bounds
 x_poison_num[1,1] <= 1
 x_poison_num[1,2] <= 1
 x_poison_num[1,3] <= 1
 x_poison_num[1,4] <= 1
 x_poison_num[1,5] <= 1
 x_poison_num[1,6] <= 1
 x_poison_num[1,7] <= 1
 x_poison_num[1,8] <= 1
 x_poison_num[1,9] <= 1
 x_poison_num[1,10] <= 1
 -10 <= weights_num[1] <= 10
 -10 <= weights_num[2] <= 10
 -10 <= weights_num[3] <= 10
 -10 <= weights_num[4] <= 10
 -10 <= weights_num[5] <= 10
 -10 <= weights_num[6] <= 10
 -10 <= weights_num[7] <= 10
 -10 <= weights_num[8] <= 10
 -10 <= weights_num[9] <= 10
 -10 <= weights_num[10] <= 10
 -10 <= bias <= 10
 -10 <= tnn_ln_times_numsamples[1,1,1] <= 10
 -10 <= tnn_ln_times_numsamples[1,1,2] <= 10
 -10 <= tnn_ln_times_numsamples[1,1,3] <= 10
 -10 <= tnn_ln_times_numsamples[1,1,4] <= 10
 -10 <= tnn_ln_times_numsamples[1,1,5] <= 10
 -10 <= tnn_ln_times_numsamples[1,1,6] <= 10
 -10 <= tnn_ln_times_numsamples[1,1,7] <= 10
 -10 <= tnn_ln_times_numsamples[1,1,8] <= 10
 -10 <= tnn_ln_times_numsamples[1,1,9] <= 10
 -10 <= tnn_ln_times_numsamples[1,1,10] <= 10
 -10 <= tnn_ln_times_numsamples[1,2,1] <= 10
 -10 <= tnn_ln_times_numsamples[1,2,2] <= 10
 -10 <= tnn_ln_times_numsamples[1,2,3] <= 10
 -10 <= tnn_ln_times_numsamples[1,2,4] <= 10
 -10 <= tnn_ln_times_numsamples[1,2,5] <= 10
 -10 <= tnn_ln_times_numsamples[1,2,6] <= 10
 -10 <= tnn_ln_times_numsamples[1,2,7] <= 10
 -10 <= tnn_ln_times_numsamples[1,2,8] <= 10
 -10 <= tnn_ln_times_numsamples[1,2,9] <= 10
 -10 <= tnn_ln_times_numsamples[1,2,10] <= 10
 -10 <= tnn_ln_times_numsamples[1,3,1] <= 10
 -10 <= tnn_ln_times_numsamples[1,3,2] <= 10
 -10 <= tnn_ln_times_numsamples[1,3,3] <= 10
 -10 <= tnn_ln_times_numsamples[1,3,4] <= 10
 -10 <= tnn_ln_times_numsamples[1,3,5] <= 10
 -10 <= tnn_ln_times_numsamples[1,3,6] <= 10
 -10 <= tnn_ln_times_numsamples[1,3,7] <= 10
 -10 <= tnn_ln_times_numsamples[1,3,8] <= 10
 -10 <= tnn_ln_times_numsamples[1,3,9] <= 10
 -10 <= tnn_ln_times_numsamples[1,3,10] <= 10
 -10 <= tnn_ln_times_numsamples[1,4,1] <= 10
 -10 <= tnn_ln_times_numsamples[1,4,2] <= 10
 -10 <= tnn_ln_times_numsamples[1,4,3] <= 10
 -10 <= tnn_ln_times_numsamples[1,4,4] <= 10
 -10 <= tnn_ln_times_numsamples[1,4,5] <= 10
 -10 <= tnn_ln_times_numsamples[1,4,6] <= 10
 -10 <= tnn_ln_times_numsamples[1,4,7] <= 10
 -10 <= tnn_ln_times_numsamples[1,4,8] <= 10
 -10 <= tnn_ln_times_numsamples[1,4,9] <= 10
 -10 <= tnn_ln_times_numsamples[1,4,10] <= 10
 -10 <= tnn_ln_times_numsamples[1,5,1] <= 10
 -10 <= tnn_ln_times_numsamples[1,5,2] <= 10
 -10 <= tnn_ln_times_numsamples[1,5,3] <= 10
 -10 <= tnn_ln_times_numsamples[1,5,4] <= 10
 -10 <= tnn_ln_times_numsamples[1,5,5] <= 10
 -10 <= tnn_ln_times_numsamples[1,5,6] <= 10
 -10 <= tnn_ln_times_numsamples[1,5,7] <= 10
 -10 <= tnn_ln_times_numsamples[1,5,8] <= 10
 -10 <= tnn_ln_times_numsamples[1,5,9] <= 10
 -10 <= tnn_ln_times_numsamples[1,5,10] <= 10
 -10 <= tnn_ln_times_numsamples[1,6,1] <= 10
 -10 <= tnn_ln_times_numsamples[1,6,2] <= 10
 -10 <= tnn_ln_times_numsamples[1,6,3] <= 10
 -10 <= tnn_ln_times_numsamples[1,6,4] <= 10
 -10 <= tnn_ln_times_numsamples[1,6,5] <= 10
 -10 <= tnn_ln_times_numsamples[1,6,6] <= 10
 -10 <= tnn_ln_times_numsamples[1,6,7] <= 10
 -10 <= tnn_ln_times_numsamples[1,6,8] <= 10
 -10 <= tnn_ln_times_numsamples[1,6,9] <= 10
 -10 <= tnn_ln_times_numsamples[1,6,10] <= 10
 -10 <= tnn_ln_times_numsamples[1,7,1] <= 10
 -10 <= tnn_ln_times_numsamples[1,7,2] <= 10
 -10 <= tnn_ln_times_numsamples[1,7,3] <= 10
 -10 <= tnn_ln_times_numsamples[1,7,4] <= 10
 -10 <= tnn_ln_times_numsamples[1,7,5] <= 10
 -10 <= tnn_ln_times_numsamples[1,7,6] <= 10
 -10 <= tnn_ln_times_numsamples[1,7,7] <= 10
 -10 <= tnn_ln_times_numsamples[1,7,8] <= 10
 -10 <= tnn_ln_times_numsamples[1,7,9] <= 10
 -10 <= tnn_ln_times_numsamples[1,7,10] <= 10
 -10 <= tnn_ln_times_numsamples[1,8,1] <= 10
 -10 <= tnn_ln_times_numsamples[1,8,2] <= 10
 -10 <= tnn_ln_times_numsamples[1,8,3] <= 10
 -10 <= tnn_ln_times_numsamples[1,8,4] <= 10
 -10 <= tnn_ln_times_numsamples[1,8,5] <= 10
 -10 <= tnn_ln_times_numsamples[1,8,6] <= 10
 -10 <= tnn_ln_times_numsamples[1,8,7] <= 10
 -10 <= tnn_ln_times_numsamples[1,8,8] <= 10
 -10 <= tnn_ln_times_numsamples[1,8,9] <= 10
 -10 <= tnn_ln_times_numsamples[1,8,10] <= 10
 -10 <= tnn_ln_times_numsamples[1,9,1] <= 10
 -10 <= tnn_ln_times_numsamples[1,9,2] <= 10
 -10 <= tnn_ln_times_numsamples[1,9,3] <= 10
 -10 <= tnn_ln_times_numsamples[1,9,4] <= 10
 -10 <= tnn_ln_times_numsamples[1,9,5] <= 10
 -10 <= tnn_ln_times_numsamples[1,9,6] <= 10
 -10 <= tnn_ln_times_numsamples[1,9,7] <= 10
 -10 <= tnn_ln_times_numsamples[1,9,8] <= 10
 -10 <= tnn_ln_times_numsamples[1,9,9] <= 10
 -10 <= tnn_ln_times_numsamples[1,9,10] <= 10
 -10 <= tnn_ln_times_numsamples[1,10,1] <= 10
 -10 <= tnn_ln_times_numsamples[1,10,2] <= 10
 -10 <= tnn_ln_times_numsamples[1,10,3] <= 10
 -10 <= tnn_ln_times_numsamples[1,10,4] <= 10
 -10 <= tnn_ln_times_numsamples[1,10,5] <= 10
 -10 <= tnn_ln_times_numsamples[1,10,6] <= 10
 -10 <= tnn_ln_times_numsamples[1,10,7] <= 10
 -10 <= tnn_ln_times_numsamples[1,10,8] <= 10
 -10 <= tnn_ln_times_numsamples[1,10,9] <= 10
 -10 <= tnn_ln_times_numsamples[1,10,10] <= 10
 -10 <= ln_numweight_times_numsample[1,1] <= 10
 -10 <= ln_numweight_times_numsample[1,2] <= 10
 -10 <= ln_numweight_times_numsample[1,3] <= 10
 -10 <= ln_numweight_times_numsample[1,4] <= 10
 -10 <= ln_numweight_times_numsample[1,5] <= 10
 -10 <= ln_numweight_times_numsample[1,6] <= 10
 -10 <= ln_numweight_times_numsample[1,7] <= 10
 -10 <= ln_numweight_times_numsample[1,8] <= 10
 -10 <= ln_numweight_times_numsample[1,9] <= 10
 -10 <= ln_numweight_times_numsample[1,10] <= 10
 Constant = 1
End
