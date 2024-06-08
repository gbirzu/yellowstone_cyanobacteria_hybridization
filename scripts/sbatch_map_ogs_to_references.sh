#!/bin/bash -l
#SBATCH --job-name=map_ogs_to_references
#SBATCH --mail-type=END         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=gbirzu@stanford.edu # Where to send mail
#SBATCH --time=24:00:00 # Time limit hrs:min:sec (max 24 hrs on Sherlock normal queue)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --partition=hns
#SBATCH --output=slurm_map_ogs_to_references_%A.out

ml ncbi-blast+

#data_dir='../results/single-cell/sscs_pangenome/'
data_dir='../results/single-cell/sscs_pangenome_v2/'
f_og_table="${data_dir}filtered_low_copy_clustered_core_orthogroup_table.tsv"
f_updated_og_table="${data_dir}filtered_low_copy_clustered_core_mapped_orthogroup_table.tsv"
blast_dir="${data_dir}_blastn_results/"
mkdir -p ${blast_dir}
num_threads=4

# Make ref files list
refs_dir='../data/reference_genomes/'
refs_files=("${refs_dir}CP000239.genbank" "${refs_dir}CP000240.genbank")
f_ref_files="${blast_dir}synecho_ref_files.txt"
if [ -f "${f_ref_files}" ]
then
    rm -f ${f_ref_files}
fi
for f_ref in ${refs_files[*]}
do
    echo "${f_ref}" >> ${f_ref_files}
done

# Export seqs
python3 pg_map_to_reference.py -O ${blast_dir} -r ${f_ref_files} --export_ref_seqs
python3 pg_map_to_reference.py -D ${data_dir} -g ${f_og_table} -o "${blast_dir}sscs_single_copy_consensus_seqs.fna" --export_og_consensus_seqs

# Find BLAST RBH against references
echo "Making SSCS consensus seqs database..."
makeblastdb -dbtype nucl -out "${blast_dir}sscs_single_copy_consensus_seqs.blastdb" -in "${blast_dir}sscs_single_copy_consensus_seqs.fna"

f_rbh="${blast_dir}blast_result_files.txt"
if [ -f ${f_rbh} ]
then
    rm -f ${f_rbh}
fi

for f_ref in ${refs_files[*]}
do
    # Make reference database
    ref_id=$(echo ${f_ref} | sed 's/.*\///g' | sed 's/\.genbank//g')
    f_blastdb="${blast_dir}${ref_id}_cds.blastdb"
    echo "Making ${ref_id} BLAST database..."
    makeblastdb -dbtype nucl -out ${f_blastdb} -in "${blast_dir}${ref_id}_cds.fna"

    # Run BLAST
    echo "Forward hit search..."
    f_forward="${blast_dir}sscs_single_copy-${ref_id}_blast_results.tsv"
    blastn -task blastn -num_threads ${num_threads} -word_size 6 -evalue 1E-3 \
            -outfmt '6 std qlen slen' -qcov_hsp_perc 75 \
            -db ${f_blastdb} \
            -out ${f_forward} \
            -query "${blast_dir}sscs_single_copy_consensus_seqs.fna"

    echo "Reverse hit search..."
    f_reverse="${blast_dir}${ref_id}-sscs_single_copy_blast_results.tsv"
    blastn -task blastn -num_threads ${num_threads} -word_size 6 -evalue 1E-3 \
            -outfmt '6 std qlen slen' -qcov_hsp_perc 75 \
            -db "${blast_dir}sscs_single_copy_consensus_seqs.blastdb" \
            -out ${f_reverse} \
            -query "${blast_dir}${ref_id}_cds.fna"

    f_merged="${blast_dir}${ref_id}_merged_blast_results.tsv"
    cat ${f_forward} ${f_reverse} > ${f_merged}
    echo ${f_merged} >> ${f_rbh}

    # Clean up
    rm -f ${f_forward}
    rm -f ${f_reverse}
done

# Map OG IDs to reference
python3 pg_map_to_reference.py -b ${f_rbh} -g ${f_og_table} -r ${f_ref_files} -o ${f_updated_og_table} --map_rbh


# Clean up
rm -f ${f_ref_files}
rm -f ${f_rbh}

