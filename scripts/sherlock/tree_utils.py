import skbio

def write_linkage_to_newick(linkage_matrix, id_list, output_file):
    tree = skbio.tree.TreeNode.from_linkage_matrix(linkage_matrix, id_list)
    skbio.io.write(tree, 'newick', output_file)
