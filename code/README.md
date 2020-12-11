# Code Folder 

Your code goes here. You could also replace the content of this file with something more meaningful


# Requirements
software: python3, gephi for some visualisations
python modules: numpy, networkx, matplotlib
probably already installed modules: random, collections, pickle
install modules for example with command: $ pip3 install numpy


# Usage
## General
to run a python file, open a terminal/shell and type for example

`...code/$ python3 experiments_holme2006.py`

most scripts and experiments generate files in the folder they are called in, we recommend calling it from a seperate folder, for example `/figures` type (on linux, windows should be similar)

```
/code $ mkdir figures
/code $ cd figures
code/figures $ python3 ../experiments_holme2006.py 
```

Files with prefix `script_` will run directly, while for `experiments_` or `exp_` you need to go to the end of the file and uncomment the experiment function you want, then run.


## Reproducing figures from the report
### General
Many calculations take a very long time which is why for some experiments we also provide parameters that yield faster but less accurate results.
### Doing your own experiments
A good starting point (minimal setup) can be found in `exp_WBT3D_and_General.py -> WBT_evolution() `. To use additional calculations on the model (metrics), varying arguments etc., take a look at `experiments_bots_short_runtime.py -> experiment_loop()` which is more complicated but very flexible. Many parameters are explained in the core module `model.py` containing the classes.

### Section 6.1 - Holme 2006

Plot from saved results: 
* 4: histograms -> `results_holme/plot_subresults_comm_size.py `.
* 5: Size of largest community ->  `results_holme/plot_subresults_max_S.py`

Plot directly from simulation (reduced size): `experiments_holme2006.py -> def ...testrun() `(see comments)

### Section 6.2 - Schweighöfer 2020
* 6: gephi graph visualization tool needed. gefx files generated with `exp_gephi_network_evo.py`
* 7: evolution of opinions in 3D --> `exp_WBT3D_and_General.py -> WBT_evolution` or in old code
* 8: old code


### Section 6.3 - Generalized WBT
* 9: gephi graph visualization tool needed. gefx files generated with `exp_gephi_network_evo.py`
* 10: contour WBT `script_WBT_contour`
* 11, 12: exp wbt3d and general ->  plot_generalized_graph. (gefx files)
* 13: exp wbt3d and general -> general_evolution
 

### Section 6.4
* If y-axis is Hyperpolarization H(O) or "Maximal absolute mean opinion", figures can be reproduced in experiments_bots_original.py or if you dont want to wait for 3 days in --> `experiments_bots_short_runtime.py` (see end of file for figure numbers)
* 20, A.29: $\phi,\varepsilon$ and bots contour plots --> `script_WBT_contour_bots.py`
* 22: network visualizations of different networks and bots (+presentation gifs) --> `script_bots_visualization.py`