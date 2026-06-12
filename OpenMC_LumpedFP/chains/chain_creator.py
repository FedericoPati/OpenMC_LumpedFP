import xml.etree.ElementTree as ET
import copy

# ---------------- CONFIG ---------------- #

ORIGINAL_CHAIN = "/data/user/pati_f/LFP_Thesis/library/chain_endfb80_sfr.xml"   # chain XML originale
MAPPING_FILE   = "/data/user/pati_f/LFP_Thesis/library/simplified_chain.txt"          # file con le righe tipo: "Np237 FIS 1.0 FPNp237"
OUTPUT_CHAIN   = "/data/user/pati_f/LFP_Thesis/library/custom_chain.xml"     # XML semplificato in output
NOT_HM_NUCLIDE = ['O16']
# ---------------------------------------- #

def parse_mapping_line(line):
    """Parsing line like: Np237 FIS 1.0 FPNp237"""
    parts = line.split()
    if len(parts) != 4:
        raise ValueError(f"Riga mapping non valida: {line}")
    nuclide_in, reaction_code, br_str, nuclide_out = parts
    br = float(br_str)
    return nuclide_in, reaction_code, br, nuclide_out

def load_original_chain(path):
    '''
    This function will return
    tree = the whole xml file in tree structure
    root = the root in which is divided, in this case only contained <depletion_chain>
    nuclides = for each nuclide present, will extract this nuclide as an element class with this attribute:
    .tag → tag name, to point a specific tag
    .attrib → dict of attributes
    .append() → to append, such as: new_rx = ET.Element("reaction",{"type": "", "target": "", "Q": ""});    .append(new_rx)
    .remove() → to remove
    .findall() → to look for under-tag, such as .findall("reaction") -> {'type': '(n,gamma)', 'target': 'Np238', 'Q': '5.0'}
    .attrib[...] = ... → to modify attributes
    '''
    tree = ET.parse(path)
    root = tree.getroot()
    # Dict : name_nuclide -> Element <nuclide>
    nuclides = {}
    for nuc in root.findall("nuclide"):
        name = nuc.attrib.get("name")
        if name:
            nuclides[name] = nuc
    return root, nuclides

def create_empty_nuclide(name):
    """Create a minimale nuclide if not present in the original chain. This is for LFP"""
    nuc = ET.Element("nuclide", {"name": name})
    return nuc

def find_reaction(original_nuclide, reaction_type):
    """
    Return the first <reaction> element of the given type found
    in the original nuclide. If multiple reactions of the same
    type exist, the first one is returned.
    """
    for rx in original_nuclide.findall("reaction"):
        if rx.attrib.get("type") == reaction_type:
            return rx
    return None

def find_decay(original_nuclide, target=None):
    """For a given nuclide, returns a <decay> that ends in 'target' (if given)."""
    if original_nuclide is None:
        return None
    for dec in original_nuclide.findall("decay"):
        if target is not None:
            if dec.attrib.get("target") != target:
                continue
        return dec
    return None

def update_counts(nuclide_elem):
    """Update the number of attributes: reactions= and decay_modes= for a <nuclide>."""
    n_rx = len(nuclide_elem.findall("reaction"))
    n_dec = len(nuclide_elem.findall("decay"))
    if n_rx > 0:
        nuclide_elem.attrib["reactions"] = str(n_rx)
    else:
        nuclide_elem.attrib.pop("reactions", None)
    if n_dec > 0:
        nuclide_elem.attrib["decay_modes"] = str(n_dec)
    else:
        nuclide_elem.attrib.pop("decay_modes", None)

def add_reaction(new_nuc, rx_code, nuclide_out, br, orig_nuc):
    """Add reaction or decay to the new nuclide based on the reaction code."""
    if rx_code == "fission":
        fiss_orig = orig_nuc.find("reaction[@type='fission']")
        nfy_orig = orig_nuc.find("neutron_fission_yields")
        fiss_new = copy.deepcopy(fiss_orig)
        new_nuc.append(fiss_new)
        nfy_new = copy.deepcopy(nfy_orig)
        for nfy in nfy_new.findall("fission_yields"):
            nfy.find("products").text = nuclide_out
            nfy.find("data").text = str(br)
        new_nuc.append(nfy_new)
        
    elif rx_code in ("(n,gamma)","(n,2n)"):
        rx_orig = find_reaction(orig_nuc, rx_code)
        rx_new = copy.deepcopy(rx_orig)
        rx_new.attrib["target"] = nuclide_out
        rx_new.attrib["branching_ratio"] = f"{br:.8g}"
        new_nuc.append(rx_new)      
              
    elif rx_code in ("alpha","beta-","ec/beta+", "IT"):
        dec_orig = find_decay(orig_nuc, target=nuclide_out)
        dec_new = copy.deepcopy(dec_orig)
        dec_new.attrib["branching_ratio"] = f"{br:.8g}"
        new_nuc.append(dec_new)        
        half_life = orig_nuc.attrib.get("half_life")
        new_nuc.set("half_life", half_life)
        decay_energy = orig_nuc.attrib.get("decay_energy")
        new_nuc.set("decay_energy", decay_energy)        
        source_photon = orig_nuc.find("source[@particle='photon']")
        new_nuc.append(copy.deepcopy(source_photon))       
        if rx_code == "alpha":
           source_alpha = orig_nuc.find("source[@particle='alpha']")
           new_nuc.append(copy.deepcopy(source_alpha))
        elif rx_code == "beta-":
              source_beta = orig_nuc.find("source[@particle='electron']")
              new_nuc.append(copy.deepcopy(source_beta))
        elif rx_code == "ec/beta+":
                source_ec = orig_nuc.find("source[@particle='positron']")
                new_nuc.append(copy.deepcopy(source_ec))
            
    else:
        raise ValueError(f"Codice reazione sconosciuto: {rx_code}")
    return new_nuc

def build_custom_chain():
    # 1) Upload original chain
    orig_root, orig_nuclides = load_original_chain(ORIGINAL_CHAIN)

    new_root = ET.Element("depletion_chain")#, attrib=orig_root.attrib)

    # 3) Dict with nuclide already created in the new file 
    new_nuclides = {}

    # 4) Set for nuclides present only as products
    product_only_nuclides = set()

    # 5) Read the input file
    with open(MAPPING_FILE, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue  # skip comments and empty lines

            nuclide_in, rx_code, br, nuclide_out = parse_mapping_line(line)

            # Indicate that the product must still appear in the final file
            product_only_nuclides.add(nuclide_out)

            # Retrieve or create the input nuclide in the new chain
            
            orig_nuc = orig_nuclides.get(nuclide_in)
            if nuclide_in not in new_nuclides.keys():
                new_nuc = create_empty_nuclide(nuclide_in)
                new_nuclides[nuclide_in] = new_nuc                              
            else:
                new_nuc = new_nuclides[nuclide_in]
            new_nuc = add_reaction(new_nuc, rx_code, nuclide_out, br, orig_nuc)    
            


    # 6) Ensure all products appear as <nuclide>
    for name in sorted(set(product_only_nuclides) | set(NOT_HM_NUCLIDE)):
        if name in new_nuclides:
            continue
        orig_nuc = orig_nuclides.get(name)
        new_nuc = create_empty_nuclide(name)
        new_nuclides[name] = new_nuc

    # 7) Aggiorna contatori reactions / decay_modes e aggiungi al root
    for nuc_name, nuc_elem in sorted(new_nuclides.items()):
        update_counts(nuc_elem)
        new_root.append(nuc_elem)

    # 8) Scrivi il nuovo XML
    new_tree = ET.ElementTree(new_root)
    ET.indent(new_tree, space="  ", level=0)
    new_tree.write(OUTPUT_CHAIN, encoding="utf-8", xml_declaration=True)
# '''
if __name__ == "__main__":
    build_custom_chain()
