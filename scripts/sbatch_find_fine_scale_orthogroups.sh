#!/bin/bash -l
#SBATCH --job-name=find_finescale_clusters
#SBATCH --mail-type=END         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=gbirzu@stanford.edu # Where to send mail
#SBATCH --time=12:00:00 # Time limit hrs:min:sec (max 24 hrs on Sherlock normal queue)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --partition=hns
#SBATCH --output=slurm_find_finescale_clusters_%A.out

#data_dir='../results/tests/pangenome_construction/sscs_v3/'
data_dir='../results/single-cell/sscs_pangenome/'
output_dir="${data_dir}filtered_orthogroups/"
f_og_table="${output_dir}sscs_filtered_orthogroup_presence.tsv"
f_scog_table="${output_dir}sscs_single_copy_orthogroup_presence.tsv"

python3 pg_update_og_table.py -i ${f_og_table} -o ${f_scog_table} -n 10 --get_high_confidence_table
input_files=($(find ${data_dir}_aln_results -name "sscs_orthogroups_?_divergence_matrices.dat" | sort))
for ((i=0; i<${#input_files[*]}; i++))
do
    f_in=${input_files[$i]}
    f_updates="${output_dir}orthogroup_updates_${i}.dat"
    echo "${f_in}"
    python3 pg_find_orthogroup_clusters.py -D ${data_dir} -O ${output_dir} -d ${f_in} -o ${f_updates} -p sscs
    python3 pg_update_og_table.py -i ${f_scog_table} -u ${f_updates} -o ${f_scog_table}
    python3 pg_update_og_table.py -i ${f_og_table} -u ${f_updates} -o ${f_og_table}
done

# Clean up old sequence files
for fname in $(find ${output_dir} -name YSG*.fna)
do 
    if [[ "${fname}" =~ "-" ]]
    then 
        # If file is subcluster of some parent OG
        parent_fname=$(echo ${fname} | sed 's/-.\{2\}\.fna/.fna/g')
        if [ -f "${parent_fname}" ]
        then 
            echo "Removing ${parent_fname}..."
            rm -f ${parent_fname}
        fi
    fi
done
