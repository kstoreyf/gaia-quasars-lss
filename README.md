# gaia-quasars-lss

This repo hosts the code to construct the Quaia catalog of Gaia-unWISE quasars. 

The associated paper can be found at https://arxiv.org/abs/2306.17749. The data products are hosted at https://doi.org/10.5281/zenodo.8060755. 

Our analysis of the cross-correlation of Quaia with CMB lensing can be found at https://arxiv.org/abs/2306.17748. The data products relevant to this analysis are hosted at https://doi.org/10.5281/zenodo.8098635.

The main relevant scripts for the Quaia paper are (to be run in this order):
- [make_data_tables.py](https://github.com/kstoreyf/gaia-quasars-lss/blob/main/code/make_data_tables.py): Constructs the necessary data tables of quasars and other sources.
- [decontaminate.py](https://github.com/kstoreyf/gaia-quasars-lss/blob/main/code/decontaminate.py): Learns optimal cuts and decontaminates the sample.
- [specphotoz.py](https://github.com/kstoreyf/gaia-quasars-lss/blob/main/code/specphotoz.py): Estimates spectrophotometric redshifts for the Quaia sources.
- [make_catalogs.py](https://github.com/kstoreyf/gaia-quasars-lss/blob/main/code/make_catalogs.py): Compiles the final catalogs, with redshift info and magnitude cuts (both the working versions and the public-facing versions).
- [selection_function_map.py](https://github.com/kstoreyf/gaia-quasars-lss/blob/main/code/selection_function_map.py): Constructs the selection function based on the catalog and feature templates.
- [generate_random.py](https://github.com/kstoreyf/gaia-quasars-lss/blob/main/code/generate_random.py): Generates a random catalog based on an input selection function.

Detailed instructions for generating custom selection functions and randoms are given in the respective files. 

The main relevant notebooks are:
- [2023-06-16_paper_figures.ipynb](https://github.com/kstoreyf/gaia-quasars-lss/blob/main/notebooks/2023-06-16_paper_figures.ipynb): Creates (most of) the figures in the paper.
- [2023-06-16_paper_quantities.ipynb](https://github.com/kstoreyf/gaia-quasars-lss/blob/main/notebooks/2023-06-16_paper_quantities.ipynb): Calculates (most of) the quantities in the paper.
- [2023-06-16_catalog_comparison.ipynb](https://github.com/kstoreyf/gaia-quasars-lss/blob/main/notebooks/2023-06-16_catalog_comparison.ipynb): Creates the figures and calculates the quantities for the catalog comparison section of the paper.
- [2023-06-16_data_products.ipynb](https://github.com/kstoreyf/gaia-quasars-lss/blob/main/notebooks/2023-06-16_data_products.ipynb): Shows how to load and plot the main data products (Quaia catalog, selection function, and randoms).

For questions or issues, please open an issue here or email [k.sf@nyu.edu](mailto:k.sf@nyu.edu).
