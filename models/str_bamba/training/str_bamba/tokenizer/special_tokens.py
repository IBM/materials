STR_SPECIAL_TOKENS = {
    ### basic tokens ###
    "BOS_TOKEN": "<bos>",
    "EOS_TOKEN": "<sep>",
    "PAD_TOKEN": "<pad>",
    "MASK_TOKEN": "<mask>",
    "UNK_TOKEN": "<unk>",

    ### molecular representations ###
    # molecular formula
    "MOLECULAR_FORMULA_TOKEN": "<formula>",

    # canonical SMILES
    "SMILES_TOKEN": "<smiles>",

    # IUPAC name
    "IUPAC_TOKEN": "<iupac>",

    # InChI
    "INCHI_TOKEN": "<inchi>",
    "INCHI_INITIAL_TOKEN": "InChI=",  # force `InChI=` to be a unique token
    "INCHI_COMMA_TOKEN": ",",  # force `,` to be a unique token
    "INCHI_DASH_TOKEN": "-",  # force `-` to be a unique token
    "INCHI_FORWARDSLASH_TOKEN": "/",  # force `/` to be a unique token
    "INCHI_QUESTIONMARK_TOKEN": "?",  # force `?` to be a unique token
    "INCHI_PARENTHESIS_OPEN_TOKEN": "(",  # force `(` to be a unique token
    "INCHI_PARENTHESIS_CLOSE_TOKEN": ")",  # force `)` to be a unique token

    # SELFIES
    "SELFIES_TOKEN": "<selfies>",

    # polymer SPG
    "POLYMER_SPG_TOKEN": "<polymer_spg>",
    "POLYMER_ARROW_TOKEN": "->",  # force `->` to be a unique token

    # formulation
    "FORMULATION_START_TOKEN": "<formulation_start>",
    "FORMULATION_END_TOKEN": "<formulation_end>",
}