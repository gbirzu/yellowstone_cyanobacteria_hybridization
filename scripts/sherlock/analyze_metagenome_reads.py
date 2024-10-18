import argparse
import re
import numpy as np
import pandas as pd
import pickle
import utils
import seq_processing_utils as seq_utils
import alignment_tools as align_utils
import pangenome_utils as pg_utils
#import matplotlib.pyplot as plt
from pangenome_utils import PangenomeMap
from metadata_map import MetadataMap
#from plot_utils import *

def strip_sample_id(sample_id):
    if 'HotsprSample' in sample_id:
        stripped_id = sample_id.replace('HotsprSample', '')
    elif 'Hotspr20Sample' in sample_id:
        stripped_id = sample_id.replace('Hotspr20Sample', '')
    elif 'HotsprSamp' in sample_id:
        stripped_id = sample_id.replace('HotsprSamp', '')
    elif 'Hotspr' in sample_id:
        stripped_id = sample_id.replace('Hotspr', '')
    else:
        stripped_id = sample_id
    stripped_id = stripped_id.replace('_FD', '')
    return stripped_id

def strip_target_id(target_id):
    return '_'.join(target_id.split('_')[:2])

def plot_total_counts(gamma_alleles_df, savefig, across='samples', num_targets=1, ylabel='reads per target', color='tab:green', label='$\gamma$', marker='D'):
    cmap = plt.get_cmap('tab10')
    fig = plt.figure(figsize=(double_col_width, single_col_width))
    ax = fig.add_subplot(111)

    sample_columns = gamma_alleles_df.columns[:-1]
    if across == 'samples':
        x_labels = [strip_sample_id(sid) for sid in sample_columns]
        read_counts = gamma_alleles_df[sample_columns].sum(axis=0)
        xlabel_fontsize = 8
        #x = np.arange(len(read_counts))
        #y = read_counts.values / num_targets
        #ax.set_xticklabels([x_labels[i] for i in x], fontsize=8, rotation=90)
    elif across == 'loci':
        #x_labels = [strip_target_id(target_id) for target_id in gamma_alleles_df.index]
        x_labels = []
        read_counts = gamma_alleles_df[sample_columns].sum(axis=1)
        xlabel_fontsize = 4

    x = np.arange(len(read_counts))
    y = read_counts.values / num_targets
    ax.set_xticks(x)
    #ax.set_xticklabels([x_labels[i] for i in x], fontsize=6, rotation=90)
    ax.set_xticklabels(x_labels, fontsize=6, rotation=90)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    #ax.set_ylim(0, 1.05)
    ax.set_ylim(9E-1, 1.5 * max(y))
    markers = ['o', 's', 'D']
    if across == 'samples':
        ax.plot(x, y, f'-{marker}', c=color, label=label)
    else:
        ax.plot(x, y, c=color, label=label)
    #for i in range(y.shape[0]):
    #    ax.plot(x, y[i], f'-{markers[i]}', c=cmap(i), label='$\gamma$')
    #    ax.errorbar(x, y[i], yerr=yerr[i], c=cmap(i))
    ax.legend()
    plt.tight_layout()
    #plt.savefig(f'{figures_dir}{locus_id}_strain_composition_per_sample.pdf')

    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    else:
        return ax


def plot_abundant_target_counts(gamma_alleles_df, savefig, ylabel='reads recruited', num_targets=10, lw=1.0, alpha=1.0):
    sample_columns = gamma_alleles_df.columns[:-1]
    x_labels = [strip_sample_id(sid) for sid in sample_columns]
    x = np.arange(len(sample_columns))

    fig = plt.figure(figsize=(double_col_width, single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xticks(x)
    ax.set_xticklabels([x_labels[i] for i in x], fontsize=8, rotation=90)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.set_ylim(9E-1, 1.5 * np.nanmax(gamma_alleles_df[sample_columns].values))
    for i in range(num_targets):
        ax.plot(x, gamma_alleles_df.iloc[i, :-1], lw=lw, alpha=alpha, label=strip_target_id(gamma_alleles_df.index[i]))
    #ax.legend()
    plt.tight_layout()
    plt.savefig(savefig)
    plt.close()


def gather_sag_gene_seqs(gene_id_subtable, seqs_dir, species=''):
    sag_seq_records = {}
    sag_id = gene_id_subtable.columns[-1]
    for sog_id in gene_id_subtable.index:
        seq_records = seq_utils.read_seqs(f'{seqs_dir}{sog_id}.fna')
        for rec in seq_records:
            if rec.id == gene_id_subtable.loc[sog_id, sag_id]:
                og_id = gene_id_subtable.loc[sog_id, 'parent_og_id']
                rec.id = f'{og_id}_alleles_{rec.id}_{species}_allele'
                #sag_seq_records.append(rec)
                sag_seq_records[rec.id] = rec
    return sag_seq_records


if __name__ == '__main__':
    default_pangenome_dir = '../results/single-cell/sscs_pangenome/'
    default_orthogroup_table_file = f'{default_pangenome_dir}filtered_orthogroups/sscs_annotated_single_copy_orthogroup_presence.tsv'
    default_metagenome_dir = '../data/metagenome/recruitment_v2/'
    default_figures_dir = '../figures/analysis/metagenome_recruitment/'

    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--figures_dir', default=default_figures_dir, help='Directory metagenome recruitment files.')
    parser.add_argument('-M', '--metagenome_dir', default=default_metagenome_dir, help='Directory metagenome recruitment files.')
    parser.add_argument('-O', '--output_dir', help='Directory in which results are saved.')
    parser.add_argument('-P', '--pangenome_dir', default=default_pangenome_dir, help='Directory with pangenome files.')
    parser.add_argument('-g', '--orthogroup_table', default=default_orthogroup_table_file, help='File with orthogroup table.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    parser.add_argument('--version', default=1, type=int, help='Analysis pipeline version.')
    args = parser.parse_args()

    pangenome_map = PangenomeMap(f_orthogroup_table=args.orthogroup_table)
    metadata = MetadataMap()

    if args.version == 1:

        # Plot gamma summary statistics
        gamma_alleles_df = pd.read_csv(f'{args.metagenome_dir}C_alleles/Callele_matches.tsv', sep='\t', index_col=0)
        metagenome_target_sites = pd.read_csv(f'{args.metagenome_dir}metagenome_recruitment_target_sites.tsv', sep='\t', index_col=0)
        target_fname_list = [strip_target_id(target_id) for target_id in np.unique(metagenome_target_sites.loc[metagenome_target_sites['notes']=='C orthogroup cluster alleles', :].index)]
        plot_total_counts(gamma_alleles_df, f'{args.figures_dir}gamma_mapped_reads_per_sample.pdf', across='samples', num_targets=len(target_fname_list))
        plot_total_counts(gamma_alleles_df, f'{args.figures_dir}gamma_mapped_reads_per_locus.pdf', across='loci', num_targets=gamma_alleles_df.shape[1] - 1)
        plot_abundant_target_counts(gamma_alleles_df, f'{args.figures_dir}gamma_most_abundant_loci.pdf', num_targets=100, lw=0.75, alpha=1)


        # Make gamma control targets
        sag_ids = pangenome_map.get_sag_ids()
        species_sorted_sag_ids = metadata.sort_sags(sag_ids, by='species')

        og_table = pangenome_map.og_table
        syna_subtable = og_table.loc[og_table['parent_og_id'].isin(target_fname_list) & og_table['sequence_cluster'].isin(['A', 'a', 'M']), :]
        syna_test_sag = syna_subtable[species_sorted_sag_ids['A']].notna().sum(axis=0).sort_values().index[-1] # choose A SAG with highest coverage at homologous SOGs
        syna_gene_seqs = gather_sag_gene_seqs(syna_subtable[['parent_og_id', syna_test_sag]].dropna(), f'{args.pangenome_dir}filtered_orthogroups/', species='A')
        seq_utils.write_seqs_dict(syna_gene_seqs, f'{args.output_dir}{syna_test_sag}_alleles.fna')

        synbp_subtable = og_table.loc[og_table['parent_og_id'].isin(target_fname_list) & og_table['sequence_cluster'].isin(['Bp', 'b', 'M']), :]
        synbp_test_sag = synbp_subtable[species_sorted_sag_ids['Bp']].notna().sum(axis=0).sort_values().index[-1] # choose B' SAG with highest coverage at homologous SOGs
        synbp_gene_seqs = gather_sag_gene_seqs(synbp_subtable[['parent_og_id', synbp_test_sag]].dropna(), f'{args.pangenome_dir}filtered_orthogroups/', species='Bp')
        print(synbp_gene_seqs, len(synbp_gene_seqs))
        seq_utils.write_seqs_dict(synbp_gene_seqs, f'{args.output_dir}{synbp_test_sag}_alleles.fna')


    elif args.version == 2:
        num_loci = 100
        markers = ['o', 's', 'D']
        colors = ['tab:orange', 'tab:blue', 'tab:green']
        for i, species in enumerate(['A', 'B', 'C']):
            f_recruitment = f'{args.metagenome_dir}{species}_Allele.xlsx'
            alleles_df = pd.read_excel(f_recruitment, sheet_name=0, index_col=0)
            sample_ids = np.array([c for c in alleles_df.columns if 'Hot' in c])
            alleles_df['average_depth'] = [np.nan, np.nan] + [np.nanmean(row.astype(float)) for row in alleles_df[sample_ids].values[2:]]
            alleles_df = alleles_df.sort_values('average_depth', ascending=False)
            loci_depth_df = alleles_df.loc[alleles_df.index[:-2], np.concatenate([sample_ids, ['average_depth']])]
            print(loci_depth_df)

            plot_abundant_target_counts(loci_depth_df, f'{args.figures_dir}{species}_loci_depths_across_samples.pdf', ylabel='read depth', num_targets=len(loci_depth_df), lw=0.75, alpha=0.5)
            plot_abundant_target_counts(loci_depth_df, f'{args.figures_dir}{species}_loci_depths_most_abundant{num_loci}.pdf', ylabel='read depth', num_targets=num_loci, lw=0.75, alpha=0.5)

            if species == 'A':
                ax = plot_total_counts(loci_depth_df, None, across='samples', num_targets=num_loci, color=colors[i], label=species, marker=markers[i])
            else:
                x = np.arange(len(loci_depth_df.columns) - 1)
                y = loci_depth_df.iloc[:num_loci, :-1].mean(axis=0).values
                ax.plot(x, y, f'-{markers[i]}', c=colors[i], label=species)

        ax.set_ylabel('read depth', fontsize=14)
        ax.legend(fontsize=10)
        plt.savefig(f'{args.figures_dir}species_comparison_most_abundant{num_loci}.pdf')
        plt.close()


