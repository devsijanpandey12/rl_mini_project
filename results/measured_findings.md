# Actual experiment findings

- **Correctness:** all 13 implementation check groups and 5 complete-study checks passed. Every one of the 3,000 Taxi transitions and 500 masks matched the installed Gymnasium reference.

- **Learning speed:** Q-learning mean return moved from -97.64 in the first 500 episodes to 2.26 in the last 500. In the alpha study, alpha=0.7 had the highest first-500 return (-59.17); inspect late behavior separately.

- **Exact benchmark:** over all 300 equally weighted initial states, the minimum route averages 13.07 actions and the optimal undiscounted return is 7.93. The optimum is start-dependent, not one fixed trip length.

- **Final Q-learning policy:** greedy success was 100.00%, mean return 7.900 (SE 0.017), mean steps 13.100, and minimum-length delivery frequency 98.50%. Truncation was 0.00%; failures remain included.

- **Exploration cost:** frozen Q-learning at epsilon=.1 returned 2.14 with 0.453 illegal service attempts and 0.266 wall no-ops per trip. Greedy return was 7.90; these are different execution policies even though Q is unchanged.

- **SARSA comparison:** greedy return -22.26, success 85.60%; exploratory return -8.54 with 0.704 illegal attempts. There were 432 capped greedy trips out of 3000 evaluations. The failure inspection reproduces one such trip if present; exploratory actions can escape a frozen greedy loop. Its sampled next-action target and finite noisy estimates differ from Q-learning's greedy target. Use these measurements rather than assuming a universal ranking.

- **Masking extension:** unmasked Q-learning made 0.444 illegal attempts per late training episode; masked training made zero by construction. Masked greedy success was 100.00% and optimal-route frequency 99.33%; masked exploratory return was 6.12. The extra validity rules are supplied knowledge.

- **Exploration schedule:** fixed .3 gave late return -14.51 and 1.766 illegal attempts; decay gave 7.46 and 0.036. Lower late exploration and the learned policy both affect this comparison.

- **Q-value accuracy:** final mean absolute Q error over 400 valid decision states ×6 actions was 2.8503; mean greedy optimal-route frequency was 98.50%. Successful routes and complete-table accuracy are distinct measurements.

- **alpha under common greedy evaluation:** alpha=0.1 had the highest observed mean return (7.930); SEs across trained seed means are in the endpoint table. Success ranged 100.0%–100.0%; optimal-route frequency ranged 96.4%–100.0%. This separates learned policy quality from training exploration costs.

- **gamma under common greedy evaluation:** gamma=1 had the highest observed mean return (7.908); SEs across trained seed means are in the endpoint table. Success ranged 95.4%–100.0%; optimal-route frequency ranged 95.3%–98.9%. This separates learned policy quality from training exploration costs.

- **epsilon under common greedy evaluation:** epsilon=0.3 had the highest observed mean return (7.922); SEs across trained seed means are in the endpoint table. Success ranged 100.0%–100.0%; optimal-route frequency ranged 98.5%–99.6%. This separates learned policy quality from training exploration costs.