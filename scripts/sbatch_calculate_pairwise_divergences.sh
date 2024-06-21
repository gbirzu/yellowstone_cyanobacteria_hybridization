#!/bin/bash -l
#SBATCH --job-name=calculate_divergences
#SBATCH --mail-type=END         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=gbirzu@stanford.edu # Where to send mail
#SBATCH --time=48:00:00 # Time limit hrs:min:sec (max 24 hrs on Sherlock normal queue)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=hns
#SBATCH --output=slurm_calculate_divergences_%A.out
#SBATCH --array=1-10 	# Array range

ml python/3.6
ml py-scipy/1.4.1_py36
ml py-pandas/1.0.3_py36

#data_dir='../results/tests/pangenome_construction/sscs_v3/'
data_dir='../results/single-cell/sscs_pangenome/'
aln_dir='../results/single-cell/alignments/v2/core_ogs_cleaned/'
out_dir='../results/single-cell/sscs_pangenome_v2/pdist/'

num_jobs=10
SLURM_ARRAY_TASK_ID=1
i=$((${SLURM_ARRAY_TASK_ID} - 1))

aln_files=($(ls ${aln_dir}*.fna | grep -v 'rRNA'))
num_files=${#aln_files[*]}
chunk_size=$(echo "${num_files} / ${num_jobs} + 1" | bc)
n_start=$(($i * ${chunk_size}))
n_end=$((($i + 1)*${chunk_size}))

input_file="${data_dir}_aln_results/sscs_orthogroup_alignment_files_${i}.txt"
if [ -f "${input_file}" ]
then
    rm -f ${input_file}
fi
for ((k=${n_start}; k<${n_end}; k++))
do
    echo ${aln_files[$k]} >> ${input_file}
done

for f_aln in $(cat ${input_file})
do
    out_stem=$(echo ${f_aln} | sed 's/.*\///g' | sed 's/_aln\.fna//g')
    f_out="${out_dir}${out_stem}"
    echo ${f_aln}, ${f_out}
    python3 calculate_pairwise_divergences.py -i ${f_aln} -o ${f_out} -m pNpS
done

rm -f ${input_file}
