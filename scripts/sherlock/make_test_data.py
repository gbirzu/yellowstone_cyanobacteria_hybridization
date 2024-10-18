import os
import pandas as pd
import numpy as np
import pangenome_utils as pg_utils


if __name__ == '__main__':
    aln_dir = '../../../ongoing/cyanobacteria/results/single-cell/sscs_pangenome/_aln_results/'
    output_dir = '../results/tests/'
    f_orthogroup_table = '../results/single-cell/sscs_pangenome/filtered_orthogroups/sscs_annotated_single_copy_orthogroup_presence.tsv'
    pangenome_map = pg_utils.PangenomeMap(f_orthogroup_table=f_orthogroup_table)
    og_table = pangenome_map.og_table
    og_ids = og_table['parent_og_id'].unique()
    '''
    f_orthogroup_table = '../results/tests/sscs_filtered_orthogroup_presence.tsv'
    og_table = pd.read_csv(f_orthogroup_table, sep='\t', index_col=0, low_memory=False)

    og_id_stems = np.array([s.split('-')[0] for s in og_table.index.values])
    og_table['og_id_stem'] = og_id_stems
    og_ids = np.unique(og_id_stems)
    print(og_table)
    print(og_ids, len(og_ids))
    print(og_ids, len(og_ids))
    '''
    n = 0
    missing_ogs = []
    for og_id in og_ids:
        if os.path.exists(f'{aln_dir}{og_id}_aln.fna'):
            n += 1
        else:
            print(og_id)
            missing_ogs.append(og_id)
    print(f'Found {n}/{len(og_ids)} OG alignments.')

    '''
    '''
