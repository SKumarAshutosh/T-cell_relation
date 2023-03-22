import re

# List of T-cells, cytokines, and transcription factors extracted from relevant articles
t_cells = ['CD4', 'CD8', 'Treg', 'Th1', 'Th2', 'Th17', 'Tfh', 'Tfh1', 'Tfh2', 'Tfh17', 'Tfh1/2', 'Tfh1/17', 'Tfh2/17',
           'Treg17', 'Treg1', 'Treg1/17', 'Treg2', 'Treg2/17', 'Treg9', 'TregTh1', 'TregTh2', 'TregTh17', 'TregTh9',
           'TregTh22', 'Trm', 'iNKT', 'MAIT', 'DN', 'DP', 'SP', 'TCR', 'Vbeta', 'Vgamma', 'Vdelta', 'CCR5', 'CXCR3',
           'CXCR5', 'CD27', 'CD28', 'CD40L', 'CD45RA', 'CD45RO', 'CD62L', 'CD69', 'CD127', 'CD152', 'CD154', 'CTLA4',
           'GITR', 'ICOS', 'LAG3', 'PD1', 'TIM3']
cytokines = ['IL-1', 'IL-2', 'IL-4', 'IL-6', 'IL-10', 'IL-12', 'IL-17', 'IL-21', 'IL-22', 'IL-23', 'IL-27', 'IL-33',
             'TNF-alpha', 'IFN-gamma', 'TGF-beta', 'GM-CSF', 'M-CSF', 'CCL', 'CXCL', 'IFN', 'MIP', 'MCP', 'RANTES',
             'IP', 'LIF', 'ILC', 'ST2']
transcription_factors = ['STAT', 'NF-kappaB', 'AP-1', 'CREB', 'IRF', 'FOXP3', 'ROR-gamma', 'T-bet', 'GATA3', 'Blimp-1',
                         'Eomes', 'Id2', 'Id3', 'Tcf1', 'Batf', 'Bcl6', 'IRF4', 'AHR', 'FOSL2', 'JUNB', 'MAF', 'MAFB',
                         'NFE2', 'NFAT', 'NFKB', 'RUNX', 'TAL1', 'TCF7']


# Label function to identify T-cells, cytokines, and transcription factors
def label_function(text):
    text_lower = text.lower()

    # Check for T-cells
    for t_cell in t_cells:
        if re.search(r"\b" + t_cell.lower() + r"\b", text_lower):
            return 'T-cell'

    # Check for cytokines
    for cytokine in cytokines:
        if re.search(r"\b" + cytokine.lower() + r"\b", text_lower):
            return 'Cytokine'

    # Check for transcription factors
    for transcription_factor in transcription_factors:
        if re.search(r"\b" + transcription_factor.lower() + r"\b", text_lower):
            return 'Transcription Factor'

    # If no match is found, return None
    return None