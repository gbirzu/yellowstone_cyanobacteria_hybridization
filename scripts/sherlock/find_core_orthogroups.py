import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import pangenome_utils as pg_utils
import metadata_map as mm
from plot_utils import *

L = 2900 # approximate number of genes in genomes

def plot_sag_coverage(og_prevalence, args):
    og_coverage = og_prevalence.sum(axis=0) / L

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('orthogroups', fontsize=14)
    ax.set_ylabel('SAGs', fontsize=14)
    ax.hist(og_prevalence.sum(axis=0), bins=30)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}sag_og_number_distribution.pdf')
    plt.close()

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('genome coverage', fontsize=14)
    ax.set_ylabel('SAGs', fontsize=14)
    ax.hist(og_coverage, bins=30)
    ax.axvline(og_coverage.mean(), lw=2, c='k', label=f'mean = {og_coverage.mean():.2f}')
    ax.legend(fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}sag_og_coverage_distribution.pdf')
    plt.close()

    return og_coverage


def plot_species_og_abundances(og_prevalence, args):
    N = len(sag_ids)
    og_abundance = og_prevalence.sum(axis=1)
    x_bins = np.arange(og_abundance.max() + 2)

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('orthogroup abundance', fontsize=14)
    ax.set_ylabel('orthogroups', fontsize=14)
    ax.hist(og_abundance, bins=x_bins)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}og_abundance_distribution.pdf')
    plt.close()

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('orthogroup abundance', fontsize=14)
    ax.set_ylabel('orthogroups', fontsize=14)
    ax.set_yscale('log')
    ax.hist(og_abundance, bins=x_bins)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}og_abundance_distribution_logy.pdf')
    plt.close()

    fig = plt.figure(figsize=(single_col_width, 0.8 * single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel('orthogroup abundance', fontsize=14)
    ax.set_ylabel('orthogroups', fontsize=14)
    ax.set_yscale('log')

    y_min = 1E-3
    color_dict = {'A':'tab:orange', 'Bp':'tab:blue'}
    label_dict = {'A':r'$\alpha$', 'Bp':r'$\beta$'}
    k_max_dict = {'A':37, 'Bp':74} # manual fit to mode of distribution
    species_core_ogs = {}
    for species in ['A', 'Bp']:
        og_abundance = og_prevalence[species_sorted_sags[species]].sum(axis=1)
        x_bins = np.arange(og_abundance.max() + 2)
        ax.hist(og_abundance, bins=x_bins, label=label_dict[species], alpha=0.5, color=color_dict[species])

        n = len(species_sorted_sags[species])
        m = len(og_abundance)
        p0 = k_max_dict[species] / n
        mu = n * p0
        sigma = np.sqrt(n * p0 * (1 - p0))
        k = np.arange(n + 1)
        theory_pdf = stats.binom.pmf(k, n, p0)
        x = k[theory_pdf > y_min]
        f0 = np.sum((og_abundance > mu - 3 * sigma) & (og_abundance < mu + 3 * sigma)) / m
        y = f0 * m * theory_pdf[theory_pdf > y_min]
        ax.plot(x, y, lw=2, color=color_dict[species])
        ax.axvline(mu - 3 * sigma, lw=2, ls='--', color=color_dict[species])

        species_core_ogs[species] = og_abundance.index.values[og_abundance > mu - 3 * sigma]

        if args.verbose:
            print(species, species_core_ogs[species], len(species_core_ogs[species]))
            print('\n')

    ax.legend(fontsize=14, frameon=False)
    plt.tight_layout()
    plt.savefig(f'{args.figures_dir}og_species_abundance_distribution_logy.pdf')
    plt.close()

    return species_core_ogs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--figures_dir', default='../figures/analysis/', help='Directory metagenome recruitment files.')
    parser.add_argument('-P', '--pangenome_dir', default='../results/single-cell/sscs_pangenome_v2/', help='Directory with pangenome files.')
    parser.add_argument('-g', '--orthogroup_table', required=True, help='File with orthogroup table.')
    parser.add_argument('-o', '--output_file', required=True, help='File with output orthogroup table.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    args = parser.parse_args()

    metadata_map = mm.MetadataMap()
    og_table = pd.read_csv(args.orthogroup_table, sep='\t', index_col=0)

    # Filter OGs with >3 species clusters
    parent_ids, parent_counts = utils.sorted_unique(og_table['parent_og_id'].values)
    og_table = og_table.loc[~og_table['parent_og_id'].isin(parent_ids[parent_counts > 3]), :]

    # Construct prevalence table
    sag_ids = np.array([c for c in og_table.columns if 'Uncmic' in c])
    species_sorted_sags = metadata_map.sort_sags(sag_ids, by='species')
    og_prevalence = pd.DataFrame(0, index=og_table.index.values, columns=['parent_og_id'] + list(sag_ids))
    og_prevalence.loc[:, 'parent_og_id'] = og_table['parent_og_id'].values
    for c in sag_ids:
        og_prevalence[c] = og_table[c].str.split(';').str.len()
    og_prevalence = og_prevalence.fillna(0)
    og_prevalence = og_prevalence.groupby(['parent_og_id']).sum()

    # Make plots
    plot_sag_coverage(og_prevalence, args)
    species_core_ogs = plot_species_og_abundances(og_prevalence, args)

    # Add core OGs
    og_table.insert(2, 'core_A', '')
    A_core_idx = og_table.loc[og_table['parent_og_id'].isin(species_core_ogs['A']), :].index.values
    og_table.loc[A_core_idx, 'core_A'] = 'Yes'
    og_table.insert(3, 'core_Bp', '')
    Bp_core_idx = og_table.loc[og_table['parent_og_id'].isin(species_core_ogs['Bp']), :].index.values
    og_table.loc[Bp_core_idx, 'core_Bp'] = 'Yes'

    # Save results
    og_table.to_csv(args.output_file, sep='\t')
    core_og_table = og_table.loc[(og_table['core_A'] == 'Yes') & (og_table['core_Bp'] == 'Yes'), :]
    f_core_only = args.output_file.replace('core', 'core-only')
    core_og_table.to_csv(f_core_only, sep='\t')


    if args.verbose:
        core_parent_ids, core_parent_counts = utils.sorted_unique(core_og_table['parent_og_id'])
        print(core_parent_ids, len(core_parent_ids))
        print(core_parent_counts)
        print(len(og_table['parent_og_id'].unique()))
        print(f'Single cluster core OGs: {np.sum(core_parent_counts == 1)}')
        print(f'Multi-cluster core OGs: {np.sum(core_parent_counts > 1)}')
        print('\n\n')

        print(og_table, len(og_table.loc[og_table['core_A'] == 'Yes', :]), len(og_table.loc[og_table['core_Bp'] == 'Yes', :]))
        print(og_table.loc[og_table['core_A'] == 'Yes', :])
        print(og_table.loc[og_table['core_Bp'] == 'Yes', :])
        print(core_og_table)


