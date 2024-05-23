The data is processed as follows. First, SAGs and contigs are filtered as described in the Appendix of [Birzu et al. (2023)](https://doi.org/10.1101/2023.06.06.543983). The pangenome is then generated from the filtered contigs using the `sag_pangenome` pipeline. The following commands are then run:

1. `python3 filter_orthogroup_table.py -O PANGENOME_DIR -g RAW_ORTHOGROUP_TABLE_PATH`
2. `bash cluster_orthogroups.sh PANGENOME_DIR -g FILTERED_ORTHOGROUP_TABLE_PATH`

The variables for the most recent analysis were the following:

- `PANGENOME_DIR` = ../results/single-cell/sscs\_pangenome\_v2/
- `RAW_ORTHOGROUP_TABLE_PATH` = ../results/single-cell/sscs\_pangenome\_v2/trimmed\_orthogroup\_table.tsv
- `FILTERED_ORTHOGROUP_TABLE_PATH` = ../results/single-cell/sscs\_pangenome\_v2/filtered\_low\_copy\_orthogroup\_table.tsv

For submitting to a computing cluster `bash` can be replaced with `sbatch` (or any other submission command) in step 2.
