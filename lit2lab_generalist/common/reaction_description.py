"""Generate mapped-reaction structural descriptions shared by all reaction types."""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set

from accfg import AccFG
from rdkit import Chem

# =========================
# 0. Basic utilities
# =========================

def mol_from_smiles_with_maps(smi: str):
    """
    Create an RDKit Mol from SMILES; assign atom maps from 1 if none are present.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smi}")
    has_map = any(a.GetAtomMapNum() for a in mol.GetAtoms())
    if not has_map:
        for i, atom in enumerate(mol.GetAtoms(), start=1):
            atom.SetAtomMapNum(i)
    return mol


def build_idx_map(mol):
    """
    Returns:
      - idx_to_map: atom_idx -> atom_map
      - map_to_idx: atom_map -> atom_idx
    """
    idx_to_map = {}
    map_to_idx = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        amap = atom.GetAtomMapNum()
        if not amap:
            # mol_from_smiles_with_maps should already have assigned atom maps.
            amap = idx + 1
            atom.SetAtomMapNum(amap)
        idx_to_map[idx] = amap
        map_to_idx[amap] = idx
    return idx_to_map, map_to_idx


def compute_global_distances(mol, root_atom_map: int, map_to_idx: Dict[int, int]):
    """
    Compute the shortest bond distance from the root atom map to each atom index.
    Returns: {atom_idx: distance_int}
    """
    if root_atom_map not in map_to_idx:
        raise ValueError(f"root_atom_map={root_atom_map} is not in the molecule")
    root_idx = map_to_idx[root_atom_map]
    dist = {root_idx: 0}
    q = deque([root_idx])
    while q:
        v = q.popleft()
        dv = dist[v]
        atom = mol.GetAtomWithIdx(v)
        for nb in atom.GetNeighbors():
            w = nb.GetIdx()
            if w not in dist:
                dist[w] = dv + 1
                q.append(w)
    return dist


# =========================
# 1. Ring system analysis
# =========================

def is_benzene_ring(mol, atom_indices: List[int]) -> bool:
    """
    Check whether a ring has six aromatic carbon atoms and aromatic C-C bonds.
    """
    if len(atom_indices) != 6:
        return False
    s = set(atom_indices)
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != "C" or not atom.GetIsAromatic():
            return False
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 in s and a2 in s:
            if not bond.GetIsAromatic():
                return False
    return True


def analyze_rings(mol):
    """
    Returns:
      - ring_systems: List[dict], each containing:
          {
            "system_id": int,
            "ring_ids": [int],
            "atom_indices": [int],
            "atom_maps": [int],
          }
      - rings: List[dict], ordered by ring_id:
          {
            "ring_id": int,
            "system_id": int,
            "atom_indices": [int],
            "atom_maps": [int],
            "size": int,
            "is_benzene": bool,
          }
      - atom_to_rings: {atom_idx: [ring_id, ...]}
      - atom_to_system: {atom_idx: system_id}
    """
    ri = mol.GetRingInfo()
    atom_rings = list(ri.AtomRings())  # list of tuples of atom indices
    n_rings = len(atom_rings)

    rings = []
    for rid, ring_atoms in enumerate(atom_rings):
        ring_atoms = list(ring_atoms)
        rings.append({
            "ring_id": rid,
            "system_id": None,
            "atom_indices": ring_atoms,
            "atom_maps": [],  # Filled below.
            "size": len(ring_atoms),
            "is_benzene": is_benzene_ring(mol, ring_atoms),
        })

    # Build atom_idx -> ring_ids first.
    atom_to_rings = defaultdict(list)
    for rid, ring in enumerate(rings):
        for idx in ring["atom_indices"]:
            atom_to_rings[idx].append(rid)

    # Build a ring graph with edges between rings that share atoms.
    adj = {i: set() for i in range(n_rings)}
    for i in range(n_rings):
        si = set(rings[i]["atom_indices"])
        for j in range(i + 1, n_rings):
            sj = set(rings[j]["atom_indices"])
            if si & sj:
                adj[i].add(j)
                adj[j].add(i)

    # Connected components define ring systems.
    ring_systems = []
    visited = set()
    for i in range(n_rings):
        if i in visited:
            continue
        stack = [i]
        comp = []
        visited.add(i)
        while stack:
            k = stack.pop()
            comp.append(k)
            for nb in adj[k]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        ring_systems.append({"system_id": len(ring_systems), "ring_ids": comp})

    # Populate system_id and atom_indices.
    idx_to_map, map_to_idx = build_idx_map(mol)
    atom_to_system = {}
    for sys in ring_systems:
        sys_id = sys["system_id"]
        atom_set = set()
        for rid in sys["ring_ids"]:
            rings[rid]["system_id"] = sys_id
            atom_set.update(rings[rid]["atom_indices"])
        sys["atom_indices"] = sorted(atom_set)
        sys["atom_maps"] = [idx_to_map[i] for i in sys["atom_indices"]]

    # Populate ring atom maps and atom_to_system.
    for sys in ring_systems:
        for idx in sys["atom_indices"]:
            atom_to_system[idx] = sys["system_id"]
    for ring in rings:
        ring["atom_maps"] = [idx_to_map[i] for i in ring["atom_indices"]]

    return ring_systems, rings, atom_to_rings, atom_to_system, idx_to_map, map_to_idx


# =========================
# 2. Non-ring fragment analysis
# =========================

def analyze_non_ring_fragments(mol, atom_to_system, idx_to_map):
    """
    Find connected components of non-ring atoms and classify them as:
      - main_chains: Fragments connected to at least two distinct ring systems.
      - side_chains: All other non-ring fragments.
    Returns:
      main_chains, side_chains, chain_atom_to_info

    where chain_atom_to_info is {atom_idx: {"type": "main"/"side", "chain_id": int}}.
    """
    n = mol.GetNumAtoms()
    ring_atoms = set(atom_to_system.keys())
    non_ring_atoms = [i for i in range(n) if i not in ring_atoms]

    visited = set()
    components = []  # list of (component_atom_indices)

    for idx in non_ring_atoms:
        if idx in visited:
            continue
        comp = []
        q = deque([idx])
        visited.add(idx)
        while q:
            v = q.popleft()
            comp.append(v)
            atom = mol.GetAtomWithIdx(v)
            for nb in atom.GetNeighbors():
                w = nb.GetIdx()
                if w in visited:
                    continue
                if w in non_ring_atoms:
                    visited.add(w)
                    q.append(w)
        components.append(sorted(comp))

    # Identify the ring systems and rings connected to each component.
    main_chains = []
    side_chains = []
    chain_atom_to_info = {}

    def _connections_for_component(atom_indices):
        """
        Returns: {atom_idx: [(system_id, ring_id, ring_atom_idx), ...], ...}
        """
        conn = defaultdict(list)
        for idx in atom_indices:
            atom = mol.GetAtomWithIdx(idx)
            for nb in atom.GetNeighbors():
                nb_idx = nb.GetIdx()
                if nb_idx in ring_atoms:
                    sys_id = atom_to_system[nb_idx]
                    conn[idx].append((sys_id, None, nb_idx))
        return conn

    # Classify main and side chains.
    for comp_idx, comp_atoms in enumerate(components):
        conn = _connections_for_component(comp_atoms)
        # Collect the distinct ring systems connected to this component.
        sys_ids = set()
        for lst in conn.values():
            for (sys_id, _, _) in lst:
                sys_ids.add(sys_id)
        if len(sys_ids) >= 2:
            chain_type = "main"
            chain_id = len(main_chains)
            main_chains.append({
                "chain_id": chain_id,
                "atom_indices": comp_atoms,
                "atom_maps": [idx_to_map[i] for i in comp_atoms],
                "connections": conn,  # system_id, ring_id=None, ring_atom_idx
            })
        else:
            chain_type = "side"
            chain_id = len(side_chains)
            side_chains.append({
                "chain_id": chain_id,
                "atom_indices": comp_atoms,
                "atom_maps": [idx_to_map[i] for i in comp_atoms],
                "connections": conn,
            })
        for i in comp_atoms:
            chain_atom_to_info[i] = {"type": chain_type, "chain_id": chain_id}

    return main_chains, side_chains, chain_atom_to_info


# =========================
# 3. Functional group information
# =========================

def build_fg_index(fg_dict: Optional[Dict[str, List[List[int]]]]):
    """
    fg_dict has the following structure:
      {
        "alkyl chloride": [[29], [101]],
        "benzene": [[23,24,25,26,27,28]],
        ...
      }
    Returns:
      - fg_dict_sorted: A version with atom maps sorted within each instance.
      - atom_map_to_fg_names: {atom_map: [fg_name1, fg_name2, ...]}
    """
    if fg_dict is None:
        return {}, {}

    fg_dict_sorted = {}
    atom_map_to_fg_names = defaultdict(list)

    for fg_name, instances in fg_dict.items():
        new_instances = []
        for inst in instances:
            inst_sorted = sorted(inst)
            new_instances.append(inst_sorted)
            for amap in inst_sorted:
                atom_map_to_fg_names[amap].append(fg_name)
        fg_dict_sorted[fg_name] = new_instances

    return fg_dict_sorted, atom_map_to_fg_names


# =========================
# 4. Ring and chain overview text
# =========================

def build_ring_overview_text(ring_systems, rings, fg_dict_sorted):
    """
    [RING_OVERVIEW]
    - Ring system 0 (fused; functional groups: indoline):
        - ring 0 (benzene ring): atoms [...]
        - ring 1 (5-membered ring): atoms [...]
    ...
    """
    # Find functional groups associated with each ring system, e.g. indoline or benzene.
    system_fg_names = defaultdict(set)
    for fg_name, instances in fg_dict_sorted.items():
        for inst in instances:
            inst_set = set(inst)
            for sys in ring_systems:
                if set(sys["atom_maps"]) == inst_set:
                    system_fg_names[sys["system_id"]].add(fg_name)

    # Find functional groups associated with individual rings, e.g. benzene.
    ring_fg_names = defaultdict(set)
    for fg_name, instances in fg_dict_sorted.items():
        for inst in instances:
            inst_set = set(inst)
            for ring in rings:
                if set(ring["atom_maps"]) == inst_set:
                    ring_fg_names[ring["ring_id"]].add(fg_name)

    lines = ["[RING_OVERVIEW]"]
    for sys in ring_systems:
        sid = sys["system_id"]
        sys_rings = [r for r in rings if r["system_id"] == sid]
        # fused_tag = " (fused ring system)" if len(sys_rings) > 1 else ""
        fused_tag = " (fused ring system)" if len(sys_rings) > 1 else ""
        fg_names = sorted(system_fg_names.get(sid, []))
        if fg_names:
            # fg_str = "; functional groups: " + ", ".join(fg_names)
            fg_str = "; functional groups: " + ", ".join(fg_names)
        else:
            fg_str = ""
        # lines.append(f"- Ring system {sid}{fused_tag}{fg_str}:")
        lines.append(f"- Ring system {sid}{fused_tag}{fg_str}:")
        for ring in sys_rings:
            rid = ring["ring_id"]
            atoms = ring["atom_maps"]
            size = ring["size"]
            # tag = "benzene ring" if ring["is_benzene"] else f"{size}-membered ring"
            tag = "benzene ring" if ring["is_benzene"] else f"{size}-membered ring"
            ring_fg = sorted(ring_fg_names.get(rid, []))
            if ring_fg:
                # rf = "; contains functional groups: " + ", ".join(ring_fg)
                rf = "; contains functional groups: " + ", ".join(ring_fg)
            else:
                rf = ""
            # lines.append(f"  - ring {rid} ({tag}): atoms {atoms}{rf}")
            lines.append(f"  - ring {rid} ({tag}): atoms {atoms}{rf}")
    return "\n".join(lines)


def build_fg_overview_text(fg_dict_sorted):
    """
    [FG_OVERVIEW]
    - fg_name: N instances
        - instance 0: atoms [...]
    """
    if not fg_dict_sorted:
        # return "[FG_OVERVIEW]\n- No explicit functional group information"
        return "[FG_OVERVIEW]\n- No explicit functional group information"

    lines = ["[FG_OVERVIEW]"]
    for fg_name, instances in fg_dict_sorted.items():
        # lines.append(f"- {fg_name}: {len(instances)} instances")
        lines.append(f"- {fg_name}: {len(instances)} instances")
        for i, inst in enumerate(instances):
            # lines.append(f"  - instance {i}: atoms {inst}")
            lines.append(f"  - instance {i}: atoms {inst}")
    return "\n".join(lines)


def refine_chain_connections_with_rings(main_chains, side_chains, atom_to_rings, rings, idx_to_map):
    """
    analyze_non_ring_fragments populates system_id while leaving ring_id as None.
    Fill in ring_id using atom_to_rings and rings.
    """
    ring_atom_to_ring_ids = atom_to_rings  # {atom_idx: [ring_id,...]}

    def _fill_chain(chain):
        new_conn = defaultdict(list)
        for chain_atom_idx, lst in chain["connections"].items():
            for (sys_id, _, ring_atom_idx) in lst:
                ring_ids = ring_atom_to_ring_ids.get(ring_atom_idx, [])
                # Keep only ring IDs in this system (normally one).
                for rid in ring_ids:
                    if rings[rid]["system_id"] == sys_id:
                        new_conn[chain_atom_idx].append(
                            (sys_id, rid, ring_atom_idx)
                        )
        chain["connections"] = new_conn

    for ch in main_chains:
        _fill_chain(ch)
    for ch in side_chains:
        _fill_chain(ch)


def build_non_ring_overview_text(
    mol,
    main_chains,
    side_chains,
    atom_to_system,
    rings,
    idx_to_map,
    fg_dict_sorted,
    atom_map_to_fg_names,
):
    """
    [NON_RING_OVERVIEW]
    - Main chain 0: atoms [...], SMILES: ...
        - Contains functional groups: ...
        - Connections to rings:
            * Chain atom [a] connects to atom [3] in ring [0] of ring system 0.
    - Side chain 0: ...
    """
    lines = ["[NON_RING_OVERVIEW]"]

    def _chain_fg_names(chain_atom_maps):
        fg_set = set()
        for amap in chain_atom_maps:
            for fg_name in atom_map_to_fg_names.get(amap, []):
                fg_set.add(fg_name)
        return sorted(fg_set)

    def _chain_smiles(atom_indices):
        if not atom_indices:
            return ""
        frag = Chem.MolFragmentToSmiles(mol, atomsToUse=atom_indices, canonical=True)
        return frag

    # Main chains
    for ch in main_chains:
        chain_id = ch["chain_id"]
        atom_indices = ch["atom_indices"]
        atom_maps = ch["atom_maps"]
        smi = _chain_smiles(atom_indices)
        fg_names = _chain_fg_names(atom_maps)
        # fg_str = ", ".join(fg_names) if fg_names else "none"
        # lines.append(f"- Main chain {chain_id}: atoms {atom_maps}, SMILES: {smi}")
        # lines.append(f"  - Contains functional groups: {fg_str}")
        fg_str = ", ".join(fg_names) if fg_names else "none"
        lines.append(f"- main chain {chain_id}: atoms {atom_maps}, SMILES: {smi}")
        lines.append(f"  - containing functional groups: {fg_str}")
        # Connectivity information
        conn = ch["connections"]
        if conn:
            # lines.append("  - Connections to rings:")
            lines.append("  - Connections to rings:")
            for chain_atom_idx, lst in conn.items():
                cmap = idx_to_map[chain_atom_idx]
                for (sys_id, rid, ring_atom_idx) in lst:
                    ring_atom_map = idx_to_map[ring_atom_idx]
                    # lines.append(
                    #     f"    * Chain atom [{cmap}] connects to ring system {sys_id}, ring [{rid}], atom [{ring_atom_map}]"
                    # )
                    lines.append(
                        f"    * Chain atom [{cmap}] is connected to ring system {sys_id}, ring [{rid}], at ring atom [{ring_atom_map}]"
                    )
        else:
            # lines.append("  - Connections to rings: none")
            lines.append("  - Connections to ring systems: none")

    # Side chains
    for ch in side_chains:
        chain_id = ch["chain_id"]
        atom_indices = ch["atom_indices"]
        atom_maps = ch["atom_maps"]
        smi = _chain_smiles(atom_indices)
        fg_names = _chain_fg_names(atom_maps)
        # fg_str = ", ".join(fg_names) if fg_names else "none"
        fg_str = ", ".join(fg_names) if fg_names else "none"
        # lines.append(f"- Side chain {chain_id}: atoms {atom_maps}, SMILES: {smi}")
        # lines.append(f"  - Contains functional groups: {fg_str}")
        lines.append(f"- Side chain {chain_id}: atoms {atom_maps}, SMILES: {smi}")
        lines.append(f"  - Contains functional groups: {fg_str}")
        conn = ch["connections"]
        if conn:
            # lines.append("  - Connections to rings:")
            lines.append("  - Connections to rings: ")
            for chain_atom_idx, lst in conn.items():
                cmap = idx_to_map[chain_atom_idx]
                for (sys_id, rid, ring_atom_idx) in lst:
                    ring_atom_map = idx_to_map[ring_atom_idx]
                    # lines.append(
                    #     f"    * Chain atom [{cmap}] connects to ring system {sys_id}, ring [{rid}], atom [{ring_atom_map}]"
                    # )
                    lines.append(
                        f"    * Chain atom(s) [{cmap}] connected to ring system {sys_id}, ring [{rid}], atom(s) [{ring_atom_map}]"
                    )
        else:
            # lines.append("  - Connections to rings: none")
            lines.append("  - Connections to rings: none")

    return "\n".join(lines)


# =========================
# 5. Overall molecular connectivity
# =========================

def build_molecule_connectivity_text(
    mol,
    ring_systems,
    rings,
    main_chains,
    side_chains,
    atom_to_system,
    idx_to_map,
):
    """
    [MOLECULE_CONNECTIVITY]
    - Within ring system 0: ring 0 ↔ ring 1, shared atoms [...]
    - Ring system 0 ↔ ring system 1: connected by a direct bond
        - Connection atoms in ring system 0: [...]
        - Connection atoms in ring system 1: [...]
    - Ring system 0 ↔ ring system 2: connected through main chain 0 (...)
      - ...
    - Ring system 1 ↔ side chain 0 (...)
      - ...
    """
    lines = ["[MOLECULE_CONNECTIVITY]"]

    # 1) Shared atoms between fused rings within the same ring system.
    for sys in ring_systems:
        sys_rids = sys["ring_ids"]
        if len(sys_rids) > 1:
            # Check shared atoms for each pair of rings.
            pairs = []
            for i in range(len(sys_rids)):
                for j in range(i + 1, len(sys_rids)):
                    r1 = rings[sys_rids[i]]
                    r2 = rings[sys_rids[j]]
                    s1 = set(r1["atom_indices"])
                    s2 = set(r2["atom_indices"])
                    shared = sorted(s1 & s2)
                    if shared:
                        pairs.append((r1["ring_id"], r2["ring_id"], shared))
            if pairs:
                for (r1_id, r2_id, shared) in pairs:
                    maps_shared = [idx_to_map[i] for i in shared]
                    # lines.append(
                    #     f"- Within ring system {sys['system_id']}: ring {r1_id} ↔ ring {r2_id}, shared atoms {maps_shared}"
                    # )
                    lines.append(
                        f"- Ring system {sys['system_id']} internal: ring {r1_id} ↔ ring {r2_id}, shared atoms {maps_shared}"
                    )

    # 2) Direct bonds between ring atoms in different ring systems.
    sys_sys_bonds = defaultdict(lambda: {"a": set(), "b": set()})
    n_atoms = mol.GetNumAtoms()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        sys_i = atom_to_system.get(i)
        sys_j = atom_to_system.get(j)
        if sys_i is None or sys_j is None:
            continue
        if sys_i == sys_j:
            continue
        key = tuple(sorted((sys_i, sys_j)))
        if sys_i < sys_j:
            sys_sys_bonds[key]["a"].add(i)
            sys_sys_bonds[key]["b"].add(j)
        else:
            sys_sys_bonds[key]["a"].add(j)
            sys_sys_bonds[key]["b"].add(i)

    for (s1, s2), info in sorted(sys_sys_bonds.items()):
        a_maps = sorted(idx_to_map[i] for i in info["a"])
        b_maps = sorted(idx_to_map[i] for i in info["b"])
        # lines.append(f"- Ring system {s1} ↔ ring system {s2}: connected by a direct bond")
        # lines.append(f"  - Connection atoms in ring system {s1}: {a_maps}")
        # lines.append(f"  - Connection atoms in ring system {s2}: {b_maps}")
        lines.append(f"- Ring system {s1} ↔ ring system {s2}: connected by a direct bond")
        lines.append(f"  - Connection atoms in ring system {s1}: {a_maps}")
        lines.append(f"  - Connection atoms in ring system {s2}: {b_maps}")


    # 3) Connections between ring systems and main or side chains.
    def _chain_connectivity(chain, label):
        chain_id = chain["chain_id"]
        atom_indices = chain["atom_indices"]
        conn = chain["connections"]
        if not conn:
            return
        systems = set()
        sys_to_ring_atoms = defaultdict(set)
        for lst in conn.values():
            for (sys_id, rid, ring_atom_idx) in lst:
                systems.add(sys_id)
                sys_to_ring_atoms[sys_id].add(ring_atom_idx)
        chain_atom_maps = [idx_to_map[i] for i in atom_indices]
        for sys_id in sorted(systems):
            ring_atom_maps = sorted(idx_to_map[i] for i in sys_to_ring_atoms[sys_id])
            # lines.append(
            #     f"- Ring system {sys_id} ↔ {label} {chain_id} (atoms {chain_atom_maps})"
            # )
            # lines.append(
            #     f"  - Connection atoms in ring system {sys_id}: {ring_atom_maps}"
            # )
            lines.append(
                f"- Ring system {sys_id} ↔ {label} {chain_id} (atoms {chain_atom_maps})"
            )
            lines.append(
                f"  - Connection atoms in ring system {sys_id}: {ring_atom_maps}"
            )

    # for ch in main_chains:
    #     _chain_connectivity(ch, "main chain")
    # for ch in side_chains:
    #     _chain_connectivity(ch, "side chain")
    for ch in main_chains:
        _chain_connectivity(ch, "main chain")
    for ch in side_chains:
        _chain_connectivity(ch, "side chain")

    return "\n".join(lines)


# =========================
# 6. Viewpoint selection
# =========================

def _is_ring_substituted_carbon(mol, atom_idx: int, ring_atom_set: Set[int]) -> bool:
    """
    Check whether a ring atom is a substituted carbon:
      - The element is C.
      - It has at least one heavy-atom (non-H) neighbor outside the ring.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() != "C":
        return False
    for nb in atom.GetNeighbors():
        if nb.GetAtomicNum() == 1:
            continue
        if nb.GetIdx() not in ring_atom_set:
            return True
    return False


def build_viewpoint_order(
    mol,
    root_atom_map: int,
    ring_systems,
    rings,
    atom_to_rings,
    atom_to_system,
    main_chains,
    side_chains,
    idx_to_map,
    map_to_idx,
    atom_map_to_fg_names: Dict[int, List[str]],
    viewpoint_verbose: str = "local",
):
    """
    Build the viewpoint order (a list of atom maps) using these rules:

    1. Ring viewpoints:
          - Include only:
              * Heteroatoms (non-C) in the ring, or
              * Carbons with a heavy-atom substituent outside this ring.
          - Exclude shared atoms that belong to multiple rings.

    2. Chain viewpoints:
          - Include:
              * Functional group center atoms present in fg_dict, and
              * Chain atoms directly bonded to a ring atom.

    3. viewpoint_verbose:
          - "local": Keep viewpoints within 4 bonds of root; always include root.
          - "all": Keep all viewpoints satisfying rules 1 and 2, regardless of distance.

    Returns:
      {
        "root_atom": root_atom_map,
        "viewpoint_traversal_order": [map1, map2, ...],
        "distances": {atom_map: bond_distance_from_root, ...}
      }
    """
    n_atoms = mol.GetNumAtoms()
    root_idx = map_to_idx[root_atom_map]

    # 1) Shortest bond distances from root to all atoms.
    dist_idx = compute_global_distances(mol, root_atom_map, map_to_idx)

    # 2) Mark ring atoms and atoms shared between fused rings.
    ring_atom_set = set(atom_to_system.keys())
    atom_idx_is_shared = {}
    for idx in range(n_atoms):
        rids = atom_to_rings.get(idx, [])
        atom_idx_is_shared[idx] = len(rids) > 1  # Multiple rings imply a shared atom.

    # 3) Collect candidate viewpoints using atom indices first.
    candidate_viewpoint_indices: Set[int] = set()

    # 3a) Ring viewpoints
    #     Ring heteroatoms and substituted carbons, excluding shared ring atoms.
    for ring in rings:
        ring_indices = ring["atom_indices"]
        ring_set = set(ring_indices)
        for idx in ring_indices:
            # Skip shared ring atoms.
            if atom_idx_is_shared.get(idx, False):
                continue
            atom = mol.GetAtomWithIdx(idx)
            sym = atom.GetSymbol()
            # Heteroatoms
            if sym != "C":
                candidate_viewpoint_indices.add(idx)
                continue
            # Substituted carbons
            if _is_ring_substituted_carbon(mol, idx, ring_set):
                candidate_viewpoint_indices.add(idx)

    # 3b) Chain viewpoints: functional group centers and atoms bonded to rings.
    fg_center_indices = set()
    for amap in atom_map_to_fg_names.keys():
        idx = map_to_idx.get(amap)
        if idx is not None:
            fg_center_indices.add(idx)

    chain_atoms_all = set()
    for ch in main_chains + side_chains:
        chain_atoms_all.update(ch["atom_indices"])

    for idx in chain_atoms_all:
        if idx in fg_center_indices:
            candidate_viewpoint_indices.add(idx)
            continue
        # Chain atoms directly bonded to ring atoms.
        atom = mol.GetAtomWithIdx(idx)
        for nb in atom.GetNeighbors():
            if nb.GetIdx() in ring_atom_set:
                candidate_viewpoint_indices.add(idx)
                break

    # 3c) Always include root as a viewpoint.
    candidate_viewpoint_indices.add(root_idx)

    # 4) In local mode, keep viewpoints within max_local_dist bonds of root.
    if viewpoint_verbose == "local":
        max_local_dist = 4
        filtered = set()
        for idx in candidate_viewpoint_indices:
            d = dist_idx.get(idx)
            if d is None:
                continue
            if d <= max_local_dist:
                filtered.add(idx)
        filtered.add(root_idx)
        candidate_viewpoint_indices = filtered

    # 5) Order viewpoints by BFS starting at root.
    visited = {root_idx}
    order_idx: List[int] = []
    q = deque([root_idx])

    while q:
        v = q.popleft()
        if v in candidate_viewpoint_indices:
            order_idx.append(v)
        atom = mol.GetAtomWithIdx(v)
        for nb in atom.GetNeighbors():
            w = nb.GetIdx()
            if w not in visited:
                visited.add(w)
                q.append(w)

    # 6) Convert atom indices to atom maps.
    viewpoint_traversal_order = [idx_to_map[i] for i in order_idx]
    distances_map = {idx_to_map[i]: d for i, d in dist_idx.items()}

    return {
        "root_atom": root_atom_map,
        "viewpoint_traversal_order": viewpoint_traversal_order,
        "distances": distances_map,
    }


# =========================
# 7. Viewpoint descriptions
# =========================

def _classify_ring_distance_label(d: int, is_benzene: bool, ring_size: int) -> str:
    """
    Map the shortest distance d within a ring to a text label:
      - For benzene rings (six atoms, is_benzene=True):
              1 -> ortho, 2 -> meta, 3 -> para; otherwise use ring bond distance d.
      - For other rings:
              1 -> ortho, 2 -> meta; otherwise use ring bond distance d.
    """
    # if d == 1:
    #     return "ortho (ring distance 1)"
    # if d == 2:
    #     return "meta (ring distance 2)"
    # if d == 3 and is_benzene and ring_size == 6:
    #     return "para (ring distance 3)"
    # return f"ring bond distance {d}"
    if d == 1:
        return "ortho (ring distance 1)"
    if d == 2:
        return "meta (ring distance 2)"
    if d == 3 and is_benzene and ring_size == 6:
        return "para (ring distance 3)"
    return f"ring bond distance {d}"

def describe_viewpoint_ring(
    mol,
    center_atom_map: int,
    root_atom_map: int,
    ring_systems,
    rings,
    atom_to_rings,
    atom_to_system,
    main_chains,
    side_chains,
    chain_atom_to_info,
    idx_to_map,
    map_to_idx,
    atom_map_to_fg_names,
    distances_map,
):
    """
    Describe a single ring-centered viewpoint.
    """
    lines = []
    center_idx = map_to_idx[center_atom_map]
    atom = mol.GetAtomWithIdx(center_idx)
    sym = atom.GetSymbol()
    root_dist = distances_map.get(center_atom_map, None)

    ring_ids = atom_to_rings.get(center_idx, [])
    if not ring_ids:
        # Fallback: this case should not occur.
        lines.append(f"[VIEWPOINT {center_atom_map}]")
        # lines.append("- Type: (error) current atom is not in any ring")
        lines.append("- Type: (Error) current atom is not in any ring")
        return "\n".join(lines)

    # ring_ids should contain one ring; shared ring atoms are excluded as viewpoints.
    ring_id = ring_ids[0]
    ring = rings[ring_id]
    system_id = ring["system_id"]
    ring_system = [sys for sys in ring_systems if sys["system_id"] == system_id][0]

    ring_atoms_idx = ring["atom_indices"]
    ring_atoms_map = ring["atom_maps"]
    ring_set = set(ring_atoms_idx)
    N = len(ring_atoms_idx)
    is_benzene = ring["is_benzene"]

    # Find functional groups associated with this entire ring system.
    system_fg_names = set()
    for amap, fgs in atom_map_to_fg_names.items():
        if amap in ring_system["atom_maps"]:
            for fg in fgs:
                system_fg_names.add(fg)

    fg_here = sorted(system_fg_names)
    # fg_str = ", ".join(fg_here) if fg_here else "none"
    fg_str = ", ".join(fg_here) if fg_here else "none"

    lines.append(f"[VIEWPOINT {center_atom_map}]")
    # lines.append(f"- Type: ring-centered view")
    # lines.append(f"- Element: {sym}")
    # lines.append(f"- Ring system: {system_id}, ring: {ring_id}")
    # lines.append(f"- Atoms in this ring: {ring_atoms_map}")
    # if root_dist is not None:
    #     lines.append(f"- Bond distance to root=[{root_atom_map}]: {root_dist}")
    # else:
    #     lines.append(f"- Bond distance to root=[{root_atom_map}]: unknown")
    # lines.append(f"- Belonging functional groups: {fg_str}")
    lines.append(f"- Type: ring-centered view")
    lines.append(f"- Element: {sym}")
    lines.append(f"- Ring system: {system_id}, ring: {ring_id}")
    lines.append(f"- Atoms in this ring: {ring_atoms_map}")
    if root_dist is not None:
        lines.append(f"- Bond distance to root=[{root_atom_map}]: {root_dist}")
    else:
        lines.append(f"- Bond distance to root=[{root_atom_map}]: unknown")
    lines.append(f"- Belonging functional groups: {fg_str}")
    lines.append("")

    # Overview of the central atom's direct connections.
    neighbors = [nb.GetIdx() for nb in atom.GetNeighbors()]
    same_system_ring_neighbors = []
    other_system_ring_neighbors = []
    chain_neighbors = []

    ring_atom_all = set(atom_to_system.keys())

    for nb_idx in neighbors:
        if nb_idx in ring_atom_all:
            sys_id_nb = atom_to_system[nb_idx]
            if sys_id_nb == system_id:
                same_system_ring_neighbors.append(nb_idx)
            else:
                other_system_ring_neighbors.append(nb_idx)
        else:
            # Chain atoms
            chain_neighbors.append(nb_idx)

    def _maps(lst):
        return [idx_to_map[i] for i in sorted(lst)]

    # lines.append("- Direct connectivity overview of the central atom:")
    # if same_system_ring_neighbors:
    #     lines.append(
    #         f"  - Connected ring atoms in the same ring system: {_maps(same_system_ring_neighbors)}"
    #     )
    # else:
    #     lines.append("  - Connected ring atoms in the same ring system: none")
    # if other_system_ring_neighbors:
    #     tmp = []
    #     for nb in other_system_ring_neighbors:
    #         sys_nb = atom_to_system[nb]
    #         tmp.append(f"ring system={sys_nb}, atom [{idx_to_map[nb]}]")
    #     lines.append("  - Connected ring atoms in other ring systems: " + ", ".join(tmp))
    # else:
    #     lines.append("  - Connected ring atoms in other ring systems: none")
    lines.append("- Direct connectivity overview of the central atom:")
    if same_system_ring_neighbors:
        lines.append(
            f"  - Connected ring atoms in the same ring system: {_maps(same_system_ring_neighbors)}"
        )
    else:
        lines.append("  - Connected ring atoms in the same ring system: none")

    if other_system_ring_neighbors:
        tmp = []
        for nb in other_system_ring_neighbors:
            sys_nb = atom_to_system[nb]
            tmp.append(f"ring system={sys_nb}, atom [{idx_to_map[nb]}]")
        lines.append("  - Connected ring atoms in other ring systems: " + ", ".join(tmp))
    else:
        lines.append("  - Connected ring atoms in other ring systems: none")

    # Identify the main or side chain containing each neighboring chain atom.
    # if chain_neighbors:
    #     parts = []
    #     for nb in chain_neighbors:
    #         info = chain_atom_to_info.get(nb)
    #         if info is None:
    #             continue
    #         amap = idx_to_map[nb]
    #         if info["type"] == "main":
    #             parts.append(f"main chain {info['chain_id']} atom [{amap}]")
    #         else:
    #             parts.append(f"side chain {info['chain_id']} atom [{amap}]")
    #     if parts:
    #         lines.append("  - Connected to main/side chains: " + ", ".join(parts))
    #     else:
    #         lines.append("  - Connected to main/side chains: none")
    # else:
    #     lines.append("  - Connected to main/side chains: none")
    if chain_neighbors:
        parts = []
        for nb in chain_neighbors:
            info = chain_atom_to_info.get(nb)
            if info is None:
                continue
            amap = idx_to_map[nb]
            if info["type"] == "main":
                parts.append(f"main chain {info['chain_id']} atom [{amap}]")
            else:
                parts.append(f"side chain {info['chain_id']} atom [{amap}]")
        if parts:
            lines.append("  - Connected to main/side chains: " + ", ".join(parts))
        else:
            lines.append("  - Connected to main/side chains: none")
    else:
        lines.append("  - Connected to main/side chains: none")

    lines.append("")

    # Feature sites on the current ring.
    # Find the center atom's position in ring order.
    try:
        pos_center = ring_atoms_idx.index(center_idx)
    except ValueError:
        pos_center = 0

    feature_sites_by_label = defaultdict(list)

    # Mark shared ring atoms.
    atom_idx_is_shared = {}
    for i in ring_atoms_idx:
        rids = atom_to_rings.get(i, [])
        atom_idx_is_shared[i] = len(rids) > 1

    for idx in ring_atoms_idx:
        if idx == center_idx:
            continue
        d_forward = (ring_atoms_idx.index(idx) - pos_center) % N
        d_backward = (pos_center - ring_atoms_idx.index(idx)) % N
        d_ring = min(d_forward, d_backward)
        label = _classify_ring_distance_label(d_ring, is_benzene, N)

        a = mol.GetAtomWithIdx(idx)
        sym_j = a.GetSymbol()
        amap_j = idx_to_map[idx]
        root_dj = distances_map.get(amap_j, None)

        # Feature sites: heteroatoms, shared ring atoms, or substituted carbons.
        is_shared = atom_idx_is_shared.get(idx, False)
        is_hetero = (sym_j != "C")
        has_sub = _is_ring_substituted_carbon(mol, idx, ring_set)

        if not (is_shared or is_hetero or has_sub):
            continue

        # Check for substituents connected to another ring or to a chain.
        nb_ring_other_system = []
        nb_chain = []
        a_neighbors = [nb.GetIdx() for nb in a.GetNeighbors()]
        for nb_idx in a_neighbors:
            if nb_idx in ring_atom_all:
                if atom_to_system[nb_idx] != system_id:
                    nb_ring_other_system.append(nb_idx)
            else:
                nb_chain.append(nb_idx)

        desc_parts = []

        # if is_hetero:
        #     desc_parts.append("heteroatom")
        # else:
        #     desc_parts.append("carbon atom")

        # if is_shared:
        #     desc_parts.append("fused-ring junction site")
        if is_hetero:
            desc_parts.append("heteroatom")
        else:
            desc_parts.append("carbon atom")

        if is_shared:
            desc_parts.append("fused-ring junction site")

        # Functional groups at this atom.
        fg_here = set(atom_map_to_fg_names.get(amap_j, []))
        if fg_here:
            # desc_parts.append("Functional groups here: " + ", ".join(sorted(fg_here)))
            desc_parts.append("Functional groups here: " + ", ".join(sorted(fg_here)))

        if nb_ring_other_system:
            # Use the first connection to another ring.
            nb0 = nb_ring_other_system[0]
            sys_nb = atom_to_system[nb0]
            # Find the ring containing nb0.
            ring_ids_nb = atom_to_rings.get(nb0, [])
            rid_target = None
            for rid in ring_ids_nb:
                if rings[rid]["system_id"] == sys_nb:
                    rid_target = rid
                    break
            if rid_target is not None:
                # desc_parts.append(
                #     f"bearing a cross-ring substituent (connected to: ring system={sys_nb}, ring={rid_target}, atom [{idx_to_map[nb0]}])"
                # )
                desc_parts.append(
                    f"bearing a cross-ring substituent (connected to: ring system={sys_nb}, ring={rid_target}, atom [{idx_to_map[nb0]}])"
                )

        if nb_chain:
            # Find the first chain connected to this atom.
            for nb in nb_chain:
                info = chain_atom_to_info.get(nb)
                if info is None:
                    continue
                # if info["type"] == "main":
                #     desc_parts.append(
                #         f"bearing a chain substituent (connected to: main chain {info['chain_id']})"
                #     )
                # else:
                #     desc_parts.append(
                #         f"bearing a chain substituent (connected to: side chain {info['chain_id']})"
                #     )
                if info["type"] == "main":
                    desc_parts.append(
                        f"bearing a chain substituent (connected to: main chain {info['chain_id']})"
                    )
                else:
                    desc_parts.append(
                        f"bearing a chain substituent (connected to: side chain {info['chain_id']})"
                    )
                break

        # root_str = (
        #     f"; bond distance to root: {root_dj}"
        #     if root_dj is not None
        #     else "; bond distance to root: unknown"
        # )
        root_str = (
            f"; bond distance to root: {root_dj}"
            if root_dj is not None
            else "; bond distance to root: unknown"
        )

        # feature_sites_by_label[label].append(
        #     f"    - atom [{amap_j}], element {sym_j}, " + "; ".join(desc_parts) + root_str
        # )
        feature_sites_by_label[label].append(
            f"    - atom [{amap_j}], element {sym_j}, " + "; ".join(desc_parts) + root_str
        )

    # if feature_sites_by_label:
    #     lines.append(f"- Feature sites on this ring (relative to center atom [{center_atom_map}]):")
    #     for label in sorted(feature_sites_by_label.keys()):
    #         lines.append(f"  {label}:")
    #         for txt in feature_sites_by_label[label]:
    #             lines.append(txt)
    # else:
    #     lines.append(
    #         f"- Feature sites on this ring (relative to center atom [{center_atom_map}]): no notable heteroatoms, substituents, or fused-ring junctions"
    #     )
    if feature_sites_by_label:
        lines.append(
            f"- Feature sites on this ring (relative to center atom [{center_atom_map}]):"
        )
        for label in sorted(feature_sites_by_label.keys()):
            lines.append(f"  {label}:")
            for txt in feature_sites_by_label[label]:
                lines.append(txt)
    else:
        lines.append(
            f"- Feature sites on this ring (relative to center atom [{center_atom_map}]): "
            "no obvious heteroatoms/substituents/fused-ring shared sites"
        )

    return "\n".join(lines)


def describe_viewpoint_chain(
    mol,
    center_atom_map: int,
    root_atom_map: int,
    main_chains,
    side_chains,
    chain_atom_to_info,
    idx_to_map,
    map_to_idx,
    atom_map_to_fg_names,
    atom_to_system,
    distances_map,
):
    """
    Describe a single chain-centered viewpoint.
    """
    lines = []
    center_idx = map_to_idx[center_atom_map]
    atom = mol.GetAtomWithIdx(center_idx)
    sym = atom.GetSymbol()
    root_dist = distances_map.get(center_atom_map, None)

    info = chain_atom_to_info.get(center_idx)
    if info is None:
        # Fallback
        lines.append(f"[VIEWPOINT {center_atom_map}]")
        # lines.append("- Type: (error) current atom is not in any chain fragment")
        lines.append("- Type: (error) current atom is not in any chain fragment")
        return "\n".join(lines)

    chain_type = info["type"]
    chain_id = info["chain_id"]
    # if chain_type == "main":
    #     chain_list = main_chains
    #     label = "main chain"
    # else:
    #     chain_list = side_chains
    #     label = "side chain"
    if chain_type == "main":
        chain_list = main_chains
        label = "main chain"
    else:
        chain_list = side_chains
        label = "side chain"

    chain = None
    for ch in chain_list:
        if ch["chain_id"] == chain_id:
            chain = ch
            break

    if chain is None:
        lines.append(f"[VIEWPOINT {center_atom_map}]")
        # lines.append("- Type: (error) no matching chain fragment found")
        lines.append("- Type: (error) no corresponding chain fragment found")
        return "\n".join(lines)

    atom_indices = chain["atom_indices"]
    atom_maps = chain["atom_maps"]
    frag_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=atom_indices, canonical=True)

    # Functional groups contained in this fragment.
    fg_names = set()
    for amap in atom_maps:
        for fg in atom_map_to_fg_names.get(amap, []):
            fg_names.add(fg)
    fg_str = ", ".join(sorted(fg_names)) if fg_names else "none"

    # lines.append(f"[VIEWPOINT {center_atom_map}]")
    # lines.append(f"- Type: chain-based view")
    # lines.append(f"- Element: {sym}")
    # lines.append(f"- Fragment: {label} {chain_id}")
    # lines.append(f"- Fragment atoms: {atom_maps}")
    # lines.append(f"- Fragment SMILES: {frag_smi}")
    # if root_dist is not None:
    #     lines.append(f"- Bond distance to root=[{root_atom_map}]: {root_dist}")
    # else:
    #     lines.append(f"- Bond distance to root=[{root_atom_map}]: unknown")
    # lines.append(f"- Functional groups in fragment: {fg_str}")
    lines.append(f"[VIEWPOINT {center_atom_map}]")
    lines.append(f"- Type: chain-based view")
    lines.append(f"- Element: {sym}")
    lines.append(f"- Fragment: {label} {chain_id}")
    lines.append(f"- Fragment atoms: {atom_maps}")
    lines.append(f"- Fragment SMILES: {frag_smi}")
    if root_dist is not None:
        lines.append(f"- Bond distance to root=[{root_atom_map}]: {root_dist}")
    else:
        lines.append(f"- Bond distance to root=[{root_atom_map}]: unknown")
    lines.append(f"- Functional groups in fragment: {fg_str}")

    # Connections between the center atom and ring systems.
    neighbors = [nb.GetIdx() for nb in atom.GetNeighbors()]
    ring_neighbors = defaultdict(list)  # sys_id -> [atom_idx,...]
    for nb_idx in neighbors:
        sys_nb = atom_to_system.get(nb_idx)
        if sys_nb is not None:
            ring_neighbors[sys_nb].append(nb_idx)

    # if ring_neighbors:
    #     lines.append("- Connections between center atom and ring systems:")
    #     for sys_id, lst in sorted(ring_neighbors.items()):
    #         maps = [idx_to_map[i] for i in sorted(lst)]
    #         lines.append(
    #             f"  - Connected to ring system {sys_id} at ring atoms: {maps}"
    #         )
    # else:
    #     lines.append("- Connections between center atom and ring systems: none")
    if ring_neighbors:
        lines.append("- Connections between center atom and ring systems:")
        for sys_id, lst in sorted(ring_neighbors.items()):
            maps = [idx_to_map[i] for i in sorted(lst)]
            lines.append(
                f"  - Connected to ring system {sys_id} at ring atoms: {maps}"
            )
    else:
        lines.append("- Connections between center atom and ring systems: none")

    return "\n".join(lines)


# =========================
# 8. Main entry point
# =========================

def describe_molecule_from_root(
    smi_or_mol,
    root_atom_map_list: List[int],
    fg_dict: Optional[Dict[str, List[List[int]]]] = None,
    viewpoint_verbose: str = "local",
):
    """
    High-level interface:
      - Input: SMILES with atom maps or an existing RDKit Mol.
      - root_atom_map: Atom map of the selected reaction site or root viewpoint.
      - fg_dict: Optional functional group information (fg_name -> [[atom_map...], ...]).
      - viewpoint_verbose: "local" or "all".

    Returns text containing these sections:
      [RING_OVERVIEW]
      [NON_RING_OVERVIEW]
      [FG_OVERVIEW]
      [MOLECULE_CONNECTIVITY]
      [VIEWPOINT_ORDER]
      [VIEWPOINT ...] ...
    """
    if isinstance(smi_or_mol, str):
        mol = mol_from_smiles_with_maps(smi_or_mol)
    else:
        mol = smi_or_mol

    # Basic indices
    ring_systems, rings, atom_to_rings, atom_to_system, idx_to_map, map_to_idx = analyze_rings(mol)

    # Non-ring fragments
    main_chains, side_chains, chain_atom_to_info = analyze_non_ring_fragments(
        mol, atom_to_system, idx_to_map
    )
    refine_chain_connections_with_rings(
        main_chains, side_chains, atom_to_rings, rings, idx_to_map
    )

    # Functional group indices
    fg_dict_sorted, atom_map_to_fg_names = build_fg_index(fg_dict)

    # Overview sections
    ring_overview = build_ring_overview_text(ring_systems, rings, fg_dict_sorted)
    non_ring_overview = build_non_ring_overview_text(
        mol,
        main_chains,
        side_chains,
        atom_to_system,
        rings,
        idx_to_map,
        fg_dict_sorted,
        atom_map_to_fg_names,
    )
    fg_overview = build_fg_overview_text(fg_dict_sorted)
    molecule_conn = build_molecule_connectivity_text(
        mol,
        ring_systems,
        rings,
        main_chains,
        side_chains,
        atom_to_system,
        idx_to_map,
    )

    # Viewpoint order
    vp_lines = []
    for root_atom_map in root_atom_map_list:
        # Distance
        dist_idx = compute_global_distances(mol, root_atom_map, map_to_idx)
        distances_map = {idx_to_map[i]: d for i, d in dist_idx.items()}
        vp_info = build_viewpoint_order(
            mol,
            root_atom_map,
            ring_systems,
            rings,
            atom_to_rings,
            atom_to_system,
            main_chains,
            side_chains,
            idx_to_map,
            map_to_idx,
            atom_map_to_fg_names,
            viewpoint_verbose=viewpoint_verbose,
        )
        vp_order = vp_info["viewpoint_traversal_order"]

        vp_lines += ["[VIEWPOINT_ORDER]", f"- root_atom: [{root_atom_map}]", f"- viewpoint_traversal_order: {vp_order}"]

        # Expand each viewpoint into a description.
        for amap in vp_order:
            idx = map_to_idx[amap]
            if idx in atom_to_system:
                # Ring viewpoint
                vp_text = describe_viewpoint_ring(
                    mol,
                    center_atom_map=amap,
                    root_atom_map=root_atom_map,
                    ring_systems=ring_systems,
                    rings=rings,
                    atom_to_rings=atom_to_rings,
                    atom_to_system=atom_to_system,
                    main_chains=main_chains,
                    side_chains=side_chains,
                    chain_atom_to_info=chain_atom_to_info,
                    idx_to_map=idx_to_map,
                    map_to_idx=map_to_idx,
                    atom_map_to_fg_names=atom_map_to_fg_names,
                    distances_map=distances_map,
                )
            else:
                # Chain viewpoint
                vp_text = describe_viewpoint_chain(
                    mol,
                    center_atom_map=amap,
                    root_atom_map=root_atom_map,
                    main_chains=main_chains,
                    side_chains=side_chains,
                    chain_atom_to_info=chain_atom_to_info,
                    idx_to_map=idx_to_map,
                    map_to_idx=map_to_idx,
                    atom_map_to_fg_names=atom_map_to_fg_names,
                    atom_to_system=atom_to_system,
                    distances_map=distances_map,
                )
            vp_lines.append("")
            vp_lines.append(vp_text)

    # Join all sections.
    parts = [
        ring_overview,
        "",
        non_ring_overview,
        "",
        fg_overview,
        "",
        molecule_conn,
        "",
        "\n".join(vp_lines),
    ]
    return "\n\n".join(parts)


def _describe_mapped_reaction(mapped_reaction):
    description_texts = []
    reactants_smiles = ""
    mapped_rxn_smiles = mapped_reaction["mapped_rxn_smiles"]
    if mapped_rxn_smiles:
        changed_atom_maps = mapped_reaction["changed_atom_maps"]
        reactants_smiles = mapped_rxn_smiles.split(">>")[0]
        reactants_mol = Chem.MolFromSmiles(reactants_smiles)
        if reactants_mol is None:
            return description_texts, reactants_smiles
        atom_mapping_to_atom_index = {}
        atom_index_to_atom_mapping = {}
        for atom in reactants_mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                continue
            atom_mapping_to_atom_index[atom.GetAtomMapNum()] = atom.GetIdx()
            atom_index_to_atom_mapping[atom.GetIdx()] = atom.GetAtomMapNum()
        for atom in reactants_mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                atom.SetAtomMapNum(len(atom_index_to_atom_mapping) + 1)
                atom_mapping_to_atom_index[atom.GetAtomMapNum()] = atom.GetIdx()
                atom_index_to_atom_mapping[atom.GetIdx()] = atom.GetAtomMapNum()
        reactants_smiles = Chem.MolToSmiles(reactants_mol)
        afg = AccFG()
        try:
            fg_dict, _ = afg.run(
                reactants_smiles,
                show_atoms=True,
                show_graph=True,
                use_atom_map_num=True,
            )
        except TypeError:
            # accfg 0.1.0 returns atom indices and no longer accepts
            # use_atom_map_num; translate its output back to atom-map IDs.
            fg_index_dict = afg.run(
                reactants_smiles,
                show_atoms=True,
                show_graph=False,
                canonical=False,
            )
            fg_dict = {
                name: [
                    [atom_index_to_atom_mapping[index] for index in occurrence]
                    for occurrence in occurrences
                ]
                for name, occurrences in fg_index_dict.items()
            }

        reaction_center_atom_mapping = list(set(changed_atom_maps))
        reaction_center_atom_mapping = [r for r in reaction_center_atom_mapping if r in atom_mapping_to_atom_index]
        if len(reaction_center_atom_mapping) == 0:
            return description_texts, reactants_smiles
        # atom_mapping = random.choice(reaction_center_atom_mapping)
        fragment_mapnum = {}
        for map_num in reaction_center_atom_mapping:
            fragment_id = get_fragment_id(reactants_mol, atom_mapping_to_atom_index[map_num])
            if fragment_id not in fragment_mapnum:
                fragment_mapnum[fragment_id] = []
            fragment_mapnum[fragment_id].append(map_num)
        
        final_fragment_mapnum = {}
        for fragment_id in fragment_mapnum:
            if len(fragment_mapnum[fragment_id]) == 1:
                final_fragment_mapnum[fragment_id] = fragment_mapnum[fragment_id]
            else:
                final_fragment_mapnum[fragment_id] = []
                used_system_id = set()
                for map_num in fragment_mapnum[fragment_id]:
                    dist_idx = compute_global_distances(reactants_mol, map_num, atom_mapping_to_atom_index)
                    distances_map = {atom_index_to_atom_mapping[i]: d for i, d in dist_idx.items()}
                    ring_system_id = get_ring_system_id(reactants_mol, atom_mapping_to_atom_index[map_num])
                    system_id = get_system_id_by_ring_attachment(reactants_mol, atom_mapping_to_atom_index[map_num])
                    # print(fragment_id, map_num, ring_system_id, system_id)
                    total_sys_id = ring_system_id | system_id
                    if len(final_fragment_mapnum[fragment_id]) > 0 and len(total_sys_id) == 0:
                        if not all([distances_map[am]>4 for am in final_fragment_mapnum[fragment_id]]):
                            continue
                    if len(used_system_id & total_sys_id) == 0:
                        final_fragment_mapnum[fragment_id].append(map_num)
                    used_system_id = used_system_id | total_sys_id
                    # print(used_system_id)
        # print(fragment_mapnum)
        # print(final_fragment_mapnum)
        for fragment_id in final_fragment_mapnum:
            text = describe_molecule_from_root(
                reactants_smiles,
                root_atom_map_list=final_fragment_mapnum[fragment_id],
                fg_dict=fg_dict,
                viewpoint_verbose="local",
            )
            elements = [
                reactants_mol.GetAtomWithIdx(
                    atom_mapping_to_atom_index[atom_mapping]
                ).GetSymbol()
                for atom_mapping in final_fragment_mapnum[fragment_id]
            ]
            description_texts.append(
                f"\n\n#### Reaction center atom {final_fragment_mapnum[fragment_id]} "
                f"element {elements}\n{text}"
            )
    return description_texts, reactants_smiles

def changed_atom_maps(mapped_rxn_smiles: str) -> List[int]:
    """Infer reaction-center atom maps from changed reactant/product bonds."""
    reactants_smiles, products_smiles = mapped_rxn_smiles.split(">>", 1)

    def mapped_bonds(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Cannot parse mapped reaction SMILES: {smiles}")
        bonds = {}
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtom().GetAtomMapNum()
            end = bond.GetEndAtom().GetAtomMapNum()
            if begin and end:
                bonds[tuple(sorted((begin, end)))] = bond.GetBondTypeAsDouble()
        return bonds

    reactant_bonds = mapped_bonds(reactants_smiles)
    product_bonds = mapped_bonds(products_smiles)
    changed_maps = set()
    for atom_pair in set(reactant_bonds) | set(product_bonds):
        if reactant_bonds.get(atom_pair) != product_bonds.get(atom_pair):
            changed_maps.update(atom_pair)
    return sorted(changed_maps)


def get_fragment_id(mol: Chem.Mol, idx1: int) -> bool:
    """
    Check whether two atoms belong to the same connected fragment.
    idx1 and idx2 are RDKit atom indices, not atom map numbers.
    """
    # Get all connected fragments as tuple(tuple(atom_idx, ...), ...).
    frags = Chem.GetMolFrags(mol, asMols=False)
    atom_to_frag = {}
    for frag_id, atom_indices in enumerate(frags):
        for aidx in atom_indices:
            atom_to_frag[aidx] = frag_id

    return atom_to_frag.get(idx1)

def build_ring_system_labels(mol: Chem.Mol):
    """
    Assign each ring a ring-system ID and map atoms to their ring systems.
    Returns:
        ring_system_id_of_ring: list[int], one entry per ring
      atom_to_ring_systems: dict[int, set[int]], atom_idx -> set of system_ids
    """
    ring_info = mol.GetRingInfo()
    atom_rings = list(ring_info.AtomRings())  # list of tuples(atom_idx,...)
    n_rings = len(atom_rings)

    # Build a ring graph: rings sharing atoms belong to the same ring system.
    ring_adj = [[] for _ in range(n_rings)]
    for i in range(n_rings):
        ai = set(atom_rings[i])
        for j in range(i + 1, n_rings):
            aj = set(atom_rings[j])
            if ai & aj:  # Shared atoms indicate a fused or spiro connection.
                ring_adj[i].append(j)
                ring_adj[j].append(i)

    # Assign system_id to each ring using graph traversal.
    ring_system_id_of_ring = [-1] * n_rings
    current_sys_id = 0
    for i in range(n_rings):
        if ring_system_id_of_ring[i] != -1:
            continue
        # Start traversing a new ring system from this ring.
        stack = [i]
        ring_system_id_of_ring[i] = current_sys_id
        while stack:
            r = stack.pop()
            for nb in ring_adj[r]:
                if ring_system_id_of_ring[nb] == -1:
                    ring_system_id_of_ring[nb] = current_sys_id
                    stack.append(nb)
        current_sys_id += 1

    # Build the atom -> ring_systems mapping.
    atom_to_ring_systems = {i: set() for i in range(mol.GetNumAtoms())}
    for r_id, atoms in enumerate(atom_rings):
        sys_id = ring_system_id_of_ring[r_id]
        for aidx in atoms:
            atom_to_ring_systems[aidx].add(sys_id)

    return ring_system_id_of_ring, atom_to_ring_systems

def get_ring_system_id(mol: Chem.Mol, idx1: int) -> bool:
    """
    Check whether two atoms belong to the same fused/shared-atom ring system.
    Return False if either atom is not in a ring.
    """
    _, atom_to_ring_systems = build_ring_system_labels(mol)

    sys1 = atom_to_ring_systems.get(idx1, set())

    return sys1

def _build_atom_to_rings(mol: Chem.Mol):
    """Return the atom_idx -> [ring_id, ...] mapping and the list of rings."""
    ring_info = mol.GetRingInfo()
    atom_rings = list(ring_info.AtomRings())  # list of tuples(atom_idx,...)

    atom_to_rings = {i: [] for i in range(mol.GetNumAtoms())}
    for rid, ring in enumerate(atom_rings):
        for a in ring:
            atom_to_rings[a].append(rid)
    return atom_rings, atom_to_rings


def _attached_ring_ids(mol: Chem.Mol, atom_idx: int,
                       atom_rings, atom_to_rings):
    """
    Find the IDs of rings to which a given atom is attached:
    - For a ring atom, return the IDs of rings containing it.
    - Otherwise, use BFS to find the nearest ring atoms and return their ring IDs.
    """
    # 1. The atom is already in a ring.
    if atom_to_rings[atom_idx]:
        return set(atom_to_rings[atom_idx])

    # 2. Use BFS to find the nearest ring atoms.
    visited = {atom_idx}
    q = deque([(atom_idx, 0)])
    nearest_ring_dist = None
    result_ring_ids = set()

    while q:
        a, d = q.popleft()
        # Stop beyond the nearest ring distance once it has been found.
        if nearest_ring_dist is not None and d > nearest_ring_dist:
            break

        if atom_to_rings[a]:
            # Record the minimum distance when the first ring atom is found.
            if nearest_ring_dist is None:
                nearest_ring_dist = d
            # Collect all rings at this distance.
            result_ring_ids.update(atom_to_rings[a])
            continue

        # Otherwise, continue expanding outward.
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            nid = nb.GetIdx()
            if nid not in visited:
                visited.add(nid)
                q.append((nid, d + 1))

    return result_ring_ids


def get_system_id_by_ring_attachment(mol: Chem.Mol, idx1: int) -> bool:
    """
    Use the following definition:
    - Atoms attached to the same ring (same ring_id) belong to the same system.
    - Atoms attached to different rings belong to different systems.
    idx1 and idx2 are RDKit atom indices.
    """
    atom_rings, atom_to_rings = _build_atom_to_rings(mol)

    # A molecule without rings has no shared ring system; return an empty set.
    if not atom_rings:
        return set()

    rings1 = _attached_ring_ids(mol, idx1, atom_rings, atom_to_rings)

    return rings1


def describe_mapped_reaction(mapped_rxn_smiles: str) -> str:
    """Describe mapped reactants around atoms whose bonds change in the reaction."""
    description_parts, _ = _describe_mapped_reaction({
        "mapped_rxn_smiles": mapped_rxn_smiles,
        "changed_atom_maps": changed_atom_maps(mapped_rxn_smiles),
    })
    if not description_parts:
        raise ValueError("Failed to generate reactant descriptions")
    return "\n".join(description_parts)
