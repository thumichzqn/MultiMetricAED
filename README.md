# MultiMetricAED
Qining Zhang, Tanner Fiez, Yi Liu, Wenyang Liu, "Multi-Metric Adaptive Experimental Design Under a Fixed Budget with Validation", AISTATS 2026.

Required Library: NumPy 1.26.4, Matplotlib 3.9.1, Scipy 1.15.2

Algorithms		folder containing algorithm classes including SHRVar and baselines

main-16.py		run the experiment for 16 treatments to obtain the probability of identifying the best treatment

validation.py		run the end to end experiment for 27 treatments to obtain the passing probability against variance heterogeneity

main-128-var.py	run the experiment for 128 treatments to obtain the probability of passing, probability of identifying best treatment, and type 1 error

main-128-err.py	run the setting where no treatment is better than control

plot.ipynb			load the data and plot figures 
