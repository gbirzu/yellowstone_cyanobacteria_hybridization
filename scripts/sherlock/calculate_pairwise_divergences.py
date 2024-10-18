import argparse
import pickle
import seq_processing_utils as seq_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', default=None, help='Input alignment in FASTA format.')
    parser.add_argument('-o', '--output_stem', default=None)
    parser.add_argument('-m', '--metric', default='nucleotide_divergence', help='["nucleotide_divergence", "pNpS"]')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    aln = seq_utils.read_alignment(args.input_file)
    if args.metric == 'nucleotide_divergence':
        pdist_df = seq_utils.calculate_fast_pairwise_divergence(aln)
        with open(f'{args.output_stem}.dat', 'wb') as out_handle:
            pickle.dump(pdist_df, out_handle)

        if args.verbose:
            print(pdist_df)
            print('\n\n')

    elif args.metric == 'pNpS':
        pN_df, pS_df = seq_utils.calculate_pairwise_pNpS(aln)
        with open(f'{args.output_stem}_pN.dat', 'wb') as out_handle:
            pickle.dump(pN_df, out_handle)
        with open(f'{args.output_stem}_pS.dat', 'wb') as out_handle:
            pickle.dump(pS_df, out_handle)

        if args.verbose:
            print('pN:')
            print(pN_df)
            print('\n')
            print('pS:')
            print(pS_df)
            print('\n\n')
