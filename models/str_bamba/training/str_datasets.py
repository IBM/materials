from torch.utils.data import Dataset


class MolecularFormulaDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.str_input = 'MOLECULAR_FORMULA'
        self.str_alternatives = ['CANONICAL_SMILES', 'IUPAC_NAME', 'INCHI', 'SELFIES']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        str_input = self.dataset[idx][self.str_input]
        sequences = [self.dataset[idx][str_name] for str_name in self.str_alternatives]
        
        return {
            'sequence_a': str_input,
            'str_alternatives': sequences
        }


class SMILESDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.str_input = 'CANONICAL_SMILES'
        self.str_alternatives = ['MOLECULAR_FORMULA', 'IUPAC_NAME', 'INCHI', 'SELFIES']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        str_input = self.dataset[idx][self.str_input]
        sequences = [self.dataset[idx][str_name] for str_name in self.str_alternatives]
        
        return {
            'sequence_a': str_input,
            'str_alternatives': sequences
        }


class IUPACDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.str_input = 'IUPAC_NAME'
        self.str_alternatives = ['MOLECULAR_FORMULA', 'CANONICAL_SMILES', 'INCHI', 'SELFIES']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        str_input = self.dataset[idx][self.str_input]
        sequences = [self.dataset[idx][str_name] for str_name in self.str_alternatives]
        
        return {
            'sequence_a': str_input,
            'str_alternatives': sequences
        }
    

class InChIDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.str_input = 'INCHI'
        self.str_alternatives = ['MOLECULAR_FORMULA', 'CANONICAL_SMILES', 'IUPAC_NAME', 'SELFIES']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        str_input = self.dataset[idx][self.str_input]
        sequences = [self.dataset[idx][str_name] for str_name in self.str_alternatives]
        
        return {
            'sequence_a': str_input,
            'str_alternatives': sequences
        }
    

class SELFIESDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.str_input = 'SELFIES'
        self.str_alternatives = ['MOLECULAR_FORMULA', 'CANONICAL_SMILES', 'IUPAC_NAME', 'INCHI']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        str_input = self.dataset[idx][self.str_input]
        sequences = [self.dataset[idx][str_name] for str_name in self.str_alternatives]
        
        return {
            'sequence_a': str_input,
            'str_alternatives': sequences
        }

class PolymerSPGDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.str_input = 'POLYMER SMILES'
        self.str_alternatives = ['POLYMER SMILES']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        str_input = self.dataset[idx][self.str_input]
        sequences = [self.dataset[idx][str_name] for str_name in self.str_alternatives]
        
        return {
            'sequence_a': str_input,
            'str_alternatives': sequences
        }


class FormulationDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.str_input = 'FORMULATION'
        self.str_alternatives = ['FORMULATION']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        str_input = self.dataset[idx][self.str_input]
        sequences = [self.dataset[idx][str_name] for str_name in self.str_alternatives]
        
        return {
            'sequence_a': str_input,
            'str_alternatives': sequences
        }