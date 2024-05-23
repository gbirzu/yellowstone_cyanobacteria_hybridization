#!/bin/bash -l
#SBATCH --job-name=cluster_orthogroups
#SBATCH --mail-type=END         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=gbirzu@stanford.edu # Where to send mail
#SBATCH --time=12:00:00 # Time limit hrs:min:sec
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --partition=hns
#SBATCH --output=slurm_cluster_orthogroups_%A.out

# Read parameters
PANDIR=$1
OGTABLE=$2
usage_message="Usage: sbatch cluster_orthogroups.sh PANGENOME_DIR_PATH ORTHOGROUP_TABLE_PATH"
if [ $# -lt 2 ]; then
  echo $usage_message 1>&2
  exit
fi

output_dir="${PANDIR}orthogroup_clusters/"
mkdir -p ${output_dir}

# Cluster orthogroups
for og_id in $(tail -n +2 ${OGTABLE} | cut -f 1)
do
    echo "Clustering ${og_id}..."
    python3 cluster_orthogroups_by_species.py -O ${output_dir} -S ${PANDIR}trimmed_aln/ \
        -d ${PANDIR}pdist/${og_id}_trimmed_pdist.dat -g ${OGTABLE}
done

# Update orthogroup table
output_table=$(echo "${OGTABLE}" | sed 's/orthogroup_table\.tsv/clustered_orthogroup_table.tsv/g')
python3 update_og_table.py -U ${output_dir} -i ${OGTABLE} -o ${output_table} -t og_clusters

mv ${output_dir}*.fna ${PANDIR}trimmed_aln/
