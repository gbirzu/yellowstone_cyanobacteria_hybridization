import pandas as pd
import numpy as np
import re
import pickle
import string

metadata_file = '../data/single-cell/JGICSP_503441_SingleCell_SampleMetaData.xlsx'
sample_key_file = '../data/single-cell/sample_metadata_key.tab'

class MetadataMap:
    #def __init__(self, f_metadata=metadata_file, f_sag_key=sample_key_file, f_a_sags='../results/single-cell/osa_sag_ids.txt', f_bp_sags='../results/single-cell/osbp_sag_ids.txt', f_species_sorting='../results/single-cell/species_sorting/sscs_species_sorted_ids.dat'):
    def __init__(self, f_metadata=metadata_file, f_sag_key=sample_key_file, 
            f_a_sags='../results/single-cell/osa_sag_ids.txt', 
            f_bp_sags='../results/single-cell/osbp_sag_ids.txt', 
            f_species_sorting='../results/single-cell/species_sorting/species_sorted_filtered_sags.dat'):
        self.metadata_file = f_metadata
        self.sag_key_file = f_sag_key
        self.sag_metadata = {}
        self.read_metadata()
        self.read_sag_key()
        #self.read_sag_species(f_a_sags, f_bp_sags)
        self.read_sag_species(f_species_sorting)
        self.map_sag_metadata()

    def __repr__(self):
        return f'{self.metadata}'

    def __getitem__(self, sag_key):
        '''
        Overload [] to access metadata dict for sag.
        '''
        return self.sag_metadata[sag_key]

    def get_sag_ids(self, key_dict):
        boolean_index = True
        for key in key_dict:
            boolean_index = boolean_index & (self.sag_key[key] == key_dict[key])
        return self.sag_key.loc[boolean_index, 'sag'].values

    def read_metadata(self):
        metadata = pd.read_excel(self.metadata_file, engine='openpyxl')
        metadata = metadata.dropna(axis=0, how='all')
        metadata['Year'] = metadata['Year'].astype(int)
        self.metadata = metadata

    def read_sag_key(self):
        sag_key = pd.read_csv(self.sag_key_file, sep='\t', header=None, names=('sag', 'description'))
        sag_key['identifier'] = sag_key['description'].str.split('.').str[1]
        sag_key['spring_name'] = sag_key['identifier'].apply(self.split_year).str[0]
        sag_key['year'] = sag_key['identifier'].apply(self.split_year).str[1]

        # Convert to metadata formatting
        sag_key['spring_name'] = sag_key['spring_name'].apply(self.translate_sag_keys)
        sag_key['year'] = sag_key['year'].apply(self.translate_sag_keys)
        self.sag_key = sag_key

    def get_sample_sag_ids(self, i_sample):
        sample_metadata = self.metadata.loc[i_sample, :]
        query_dict = {'spring_name':sample_metadata['Spring_Name'], 'year':sample_metadata['Year']}
        return self.get_sag_ids(query_dict)

    def get_sample_index(self):
        return list(self.metadata.index)

    def get_sample_id(self, i_sample):
        sample_metadata = self.metadata.loc[i_sample, :]
        loc_dict = {'Mushroom Spring':'MS', 'Octopus Spring':'OS'}
        return f'{loc_dict[sample_metadata["Spring_Name"]]}{sample_metadata["Year"]}'

    @staticmethod
    def split_year(string):
        return re.split('(\d+)', string)

    def translate_sag_keys(self, key):
        location_dict = {'Mush':'Mushroom Spring', 'Octo':'Octopus Spring'}
        if key in location_dict.keys():
            formated_key = location_dict[key]
        else:
            # Assume key is year in YY format. Convert to YYYY int.
            formated_key = int('20' + key)
        return formated_key

    def read_sag_species(self, f_species_sorting):
        # Old version
        '''
        sag_species = pickle.load(open(f_species_sorting, 'rb'))
        dref_species = sag_species['d_ref']
        self.sag_species = {'A':dref_species['syna'], 'Bp':dref_species['synbp']}

        if 'hitchhiking_alleles' in sag_species:
            synbp_strains = sag_species['hitchhiking_alleles']
            self.synbp_strains = synbp_strains
        '''

        # Updated version using filtered data
        self.sag_species = pickle.load(open(f_species_sorting, 'rb'))

    def map_sag_metadata(self):
        self.sag_key['temperature'] = float('NaN')
        sag_dict = {}
        for sag in self.sag_key['sag']:
            sag_dict[sag] = self.get_metadata(sag)
            self.sag_key.loc[self.sag_key['sag'] == sag, 'temperature'] = sag_dict[sag]['temperature']
        self.sag_metadata = sag_dict

    def get_metadata(self, sag):
        identifiers = self.sag_key.loc[self.sag_key['sag'] == sag, ['spring_name', 'year']].values[0]
        if sag in self.sag_species['A']:
            sag_species = 'A'
        elif sag in self.sag_species['Bp']:
            sag_species = 'Bp'
        else:
            sag_species = None
        sag_dict = {'sag_id':sag, 'spring_name':identifiers[0], 'year':identifiers[1], 'species':sag_species}
        self.add_sag_metadata(sag_dict)
        return sag_dict

    def add_sag_metadata(self, sag_dict):
        metadata_info = self.metadata.loc[(self.metadata['Year'] == sag_dict['year']) & (self.metadata['Spring_Name'] == sag_dict['spring_name']), :]
        sag_dict['temperature'] = metadata_info['Temperature'].values[0]
        sag_dict['collection_date'] = metadata_info['Collection_Day'].values[0]
        sag_dict['collection_time'] = metadata_info['Collection_Time'].values[0]

    def sort_sags(self, sag_list, by='sample'):
        sorted_sags = {}
        if by == 'sample':
            for i, row in self.metadata.iterrows():
                sample_id = self.get_sample_id(i)
                sag_ids = self.get_sag_ids({'spring_name':row['Spring_Name'], 'year':row['Year']})
                sorted_sags[sample_id] = [sag for sag in sag_list if sag in sag_ids]
        elif by == 'location':
            location_dict = {'Mushroom Spring':'MS', 'Octopus Spring':'OS'}
            for location in location_dict:
                sag_ids = self.get_sag_ids({'spring_name':location})
                sorted_sags[location_dict[location]] = [sag for sag in sag_list if sag in sag_ids]
        elif by == 'temperature':
            temperatures = np.unique(self.metadata['Temperature'].values)
            temperatures = temperatures[np.isfinite(temperatures)]
            for temp in temperatures:
                sag_ids = self.get_sag_ids({'temperature':temp})
                sorted_sags[f'T{temp}'] = [sag for sag in sag_list if sag in sag_ids]
        elif by == 'species':
            for species in self.sag_species:
                sorted_sags[species] = [sag_id for sag_id in sag_list if sag_id in self.sag_species[species]]
        elif by == 'Bp_strains':
            for strain in self.synbp_strains:
                sorted_sags[strain] = [sag_id for sag_id in sag_list if sag_id in self.synbp_strains[strain]]
        return sorted_sags

    def get_sag_species(self, sag_id):
        sag_species = None
        for species in self.sag_species:
            if sag_id in self.sag_species[species]:
                sag_species = species
                break
        return sag_species

    def get_all_sag_ids(self):
        temp = []
        for location in ['Mushroom Spring', 'Octopus Spring']:
            temp.append(self.get_sag_ids({'spring_name':location}))
        return np.concatenate(temp)

    def get_high_confidence_sag_ids(self):
        '''
        Returns SAG IDs sorted into one of the three species bins 'A', 'Bp', and 'C'.
        '''
        temp = []
        for species in ['A', 'Bp', 'C']:
            temp.append(self.sag_species[species])
        return np.concatenate(temp)


class PlateCoordinates:
    '''
    Stores coordinates of reaction well given SAG ID.
    '''
    def __init__(self, sag_id=None):
        self.metadata = MetadataMap()
        self.initialize_plate_ids()
        self.x_row = np.array([chr(65 + i) for i in range(16)]) # A-P
        self.x_column = np.arange(1, 25, dtype=int)
        self.location_dict = {'Mushroom Spring':'MS', 'MS':'Mushroom Spring', 'Octopus Spring':'OS', 'OS':'Octopus Spring'}

        if sag_id is not None:
            self.cells = {sag_id:self.get_sag_coordinates(sag_id)}
        else:
            self.cells = {}

    def initialize_plate_ids(self):
        # Make sequencing plate vector
        sample_ids = sorted([self.metadata.get_sample_id(i) for i in self.metadata.get_sample_index()])
        x_plate = []
        for s in sample_ids:
            if s != 'OS2009':
                x_plate.append(s)
            else:
                # Separate OS2009 int A1, A2, A3 which appear to have been sequenced separately
                x_plate.append(f'{s}_A1')
                x_plate.append(f'{s}_A2')
                x_plate.append(f'{s}_A3')
        self.x_plate = np.array(x_plate)

    def get_sag_coordinates(self, sag_id):
        x = np.zeros(3, dtype=int) # (i_sample, j_row, k_column) mapping sample, rown and column on 384 well plate
        sag_metadata = self.metadata.get_metadata(sag_id)
        sample_id = f'{self.location_dict[sag_metadata["spring_name"]]}{sag_metadata["year"]}'
        sag_plate_markers = self.read_plate_markers(sag_id)
        
        # Add plate coordinate
        if sample_id != 'OS2009':
            x[0] = np.arange(len(self.x_plate))[sample_id == self.x_plate]
        else:
            os09_id = f'{sample_id}_{sag_plate_markers["A"]}'
            x[0] = np.arange(len(self.x_plate))[os09_id == self.x_plate]

        # Add row and column
        x[1] = np.arange(len(self.x_row))[self.x_row == sag_plate_markers['row']]
        x[2] = sag_plate_markers['column'] - 1

        return x

    def read_plate_markers(self, sag_id):
        id_words = re.findall('[A-Z][a-z0-9]*', re.sub('_FD$', '', sag_id)) # [prefix, spring, 'Red', 'A##', 'jk']
        markers = {'A':id_words[3]} # A label
        plate_loc = re.split('(\d+)', id_words[4].split('_')[0])
        markers['row'] = plate_loc[0]
        markers['column'] = int(plate_loc[1])
        return markers
        
    def get_neighbors(self, x):
        j = np.array([0, 1, 0], dtype=int)
        k = np.array([0, 0, 1], dtype=int)
        return np.array([x - (x[1] > 0) * j, x + (x[1] < len(self.x_row) - 1) * j, x - (x[2] > 0) * k, x + (x[2] < len(self.x_column) - 1) * k])

    def get_sag_id(self, x):
        # Get markers
        plate_id = self.x_plate[x[0]]
        if 'OS2009' in plate_id:
            sample_id, A_marker = plate_id.split('_')
        else:
            sample_id = plate_id
            A_marker = ''
        xy = f'{self.x_row[x[1]]}{self.x_column[x[2]]}'

        # Search for ID in sample
        spring, year = re.split('(\d+)', sample_id)[:2]
        sample_key = {'spring_name':self.location_dict[spring], 'year':int(year)}
        sample_sag_ids = self.metadata.get_sag_ids(sample_key)
        target_ids = [s_id for s_id in sample_sag_ids if f'{A_marker}{xy}' in s_id]
        
        if len(target_ids) == 1:
            return target_ids[0]
        elif len(target_ids) == 0:
            print(f'No SAG ID found at x={x}!')
            return None
        else:
            print(f'Multiple SAG IDs found at x={x}!')
            return target_ids


if __name__ == '__main__':
    metadata = MetadataMap(metadata_file, sample_key_file)
    print(metadata.metadata)
    print(metadata['UncmicMuRedA1C10_FD'])
    print(metadata.get_sag_ids({'spring_name':'Mushroom Spring', 'year':2006}))
    print(metadata.get_sag_ids({'temperature':59}))
