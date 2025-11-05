This repository contains the data and code to reproduce the results from 
G. Birzu, H. Subrahmaniam Muralidharan, D. Goudeau, R.R. Malmstrom, D.S. Fisher, D. Bhaya, Hybridization breaks species barriers in long-term coevolution of a cyanobacterial population ([bioRxiv:2023.06.06.543983](https://doi.org/10.1101/2023.06.06.543983)). All code is provided as is and subjecto the terms listed in the associated license agreement.

### Usage (Unix)

1. Download large data files from [Zenodo repository](https://doi.org/10.5281/zenodo.17534464) to `results/`
2. Run `tar xvzf yellowstone_sags_large_files.tar.gz` from `results/`
3. Copy large files using `cp -R main_figures_large_files/* main_figures_data/`
3. Run `python make_main_figures.py` from `scripts/`