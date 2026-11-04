"""Prompt templates for all reaction-yield workflows."""

import re
from textwrap import dedent


# Acidamine GPT-5 annotation prompt
def get_acidamine_prompt(reaction_data, reactants_description, gt):
    return dedent(f"""\
                    You are a senior organic chemist specializing in reaction mechanisms, reactivity prediction, and structure–reactivity relationships.

                    CONTEXT  
                    - You are given:  
                    • Raw reaction data (including cano_rxn_smiles / mapped_rxn_smiles, catalysts, reagents, solvent).  
                    • A detailed reaction-center annotation (ring systems, functional groups, connectivity, viewpoints, atom mapping).  

                    YOUR TASK  
                    Analyze the given reaction, identify its reaction type, and decide whether, under the given conditions, the reaction is more likely to have a high or low yield. Yield >20% is considered “high”.

                    You MUST reason explicitly about:  
                    - The local structural environment around the reaction center (rings, functional groups, sterics, electronics).  
                    - The suitability of the conditions (catalysts / reagents / solvent) for the identified reaction type.  
                    - For acid–amine amide coupling specifically (“acidamine”), whether the most important factors are present, missing, or conflicting in the reaction entry.

                    ------------------------------------------------
                        
                    INPUT

                    REACTION DATA:
                    {reaction_data}

                    REACTION-CENTER ANNOTATION:
                    {reactants_description}
                    ------------------------------------------------

                    I will privately tell you that the yield level of the following reaction is {gt}. Please act as if you do not know this information.
                    
                    ------------------------------------------------

                    OUTPUT FORMAT (VERY IMPORTANT)

                    You MUST output:

                    <think>
                    ...structured fields...
                    ...short explanation sentences...
                    </think>
                    <answer>high or low</answer>

                    No other text before or after.

                    ------------------------------------------------
                    DETAILED REQUIREMENTS FOR <think>

                    1. Reaction type identification (MANDATORY)

                    You MUST include exactly one line:

                    Reaction_type = ...

                    - If the transformation is an acid–amine amide coupling (acidamine), you MUST write:
                    Reaction_type = acid-amine amide coupling

                    - If it is clearly NOT an acid–amine amide coupling, write:
                    Reaction_type = other

                    (You may replace “other” with a more specific label like “esterification” or “nucleophilic aromatic substitution” if you can, but it MUST NOT be “acid-amine amide coupling” unless the main new bond is C(=O)–N formed between a carboxylic acid (or activated derivative) and an amine.)

                    2. Reaction-center identification (MANDATORY)

                    You MUST include exactly one line:

                    Reaction_site = [a, b, c, ...]

                    Where the list is the atom indices (from the mapped_rxn_smiles / annotation) that are directly involved in bond formation or cleavage at the main reaction center.  
                    Examples:
                    Reaction_site = [2, 4]  
                    Reaction_site = [2, 3, 4]

                    3. Acid–amine–specific structural fields

                    - If Reaction_type = acid-amine amide coupling:

                    You MUST provide the following lines (do NOT skip any):

                    Acid_atom = i
                    Amine_atom = j
                    Acid_functional_group = ...
                    Amine_functional_group = ...
                    Local_environment_key_groups = ...

                    Where:
                    - Acid_atom is the atom index of the acyl carbon (or the main electrophilic carbon in the acid/activated acyl partner).  
                    - Amine_atom is the atom index of the nitrogen that becomes the amide nitrogen.  
                    - Acid_functional_group is a short name like “carboxylic acid”, “acid chloride”, “activated ester”, “anhydride”, etc.  
                    - Amine_functional_group is a short name like “primary aliphatic amine”, “secondary aliphatic amine”, “aniline”, “amide nitrogen (weak nucleophile)”, etc.  
                    - Local_environment_key_groups should list key nearby functional groups and their atom indices, e.g.  
                        Local_environment_key_groups = acid_fg: carboxylic acid [2, 3, 39]; nucleophile_fg: secondary aliphatic amine [4]; neighbors: sulfonyl [9,10,11], secondary amide [20,21,22], tertiary alcohol [28], CF3 [29-36]

                    - If Reaction_type ≠ acid-amine amide coupling:

                    You MUST still include the same lines, but set them to NA:

                    Acid_atom = NA  
                    Amine_atom = NA  
                    Acid_functional_group = NA  
                    Amine_functional_group = NA  
                    Local_environment_key_groups = NA

                    4. Acidamine condition checklist (if and only if Reaction_type = acid-amine amide coupling)

                    If Reaction_type = acid-amine amide coupling, you MUST give the following checklist lines, each with a categorical status AND a very short justification derived ONLY from the given reaction data (catalysts / reagents / solvent / SMILES / annotation):

                    Acid_activation = STATUS (reason...)
                    Base_situation = STATUS (reason...)
                    Solvent_class = STATUS (reason...)
                    Solvent_support = STATUS (reason...)
                    Competitive_nucleophiles = [...] (short note)
                    Protecting_group_sensitivity = STATUS (reason...)

                    - Allowed STATUS values (choose the closest):

                    For Acid_activation:
                        present / likely / absent / unclear

                    For Base_situation:
                        suitable_base_present / base_missing / base_too_weak / base_too_strong / unclear

                    For Solvent_class:
                        polar_aprotic / medium_polarity / nonpolar / protic / mixed / unclear

                    For Solvent_support:
                        supportive / neutral / hindering / unclear

                    For Protecting_group_sensitivity:
                        compatible / potentially_problematic / unclear

                    - Competitive_nucleophiles:
                    - Must be a bracketed list, e.g.  
                        Competitive_nucleophiles = [22, 28] (amide NH, tertiary OH; both deactivated compared to main amine)  
                        or  
                        Competitive_nucleophiles = [] (no obvious competing nucleophiles)

                    You MUST base these statuses ONLY on information explicitly present in:
                    - catalysts / reagents / solvent strings, and  
                    - the functional group / connectivity annotation and mapped SMILES.  
                    Do NOT invent reagents or conditions that are not listed.

                    If Reaction_type ≠ acid-amine amide coupling, you MUST still include these lines, but you can set:

                    Acid_activation = NA  
                    Base_situation = NA  
                    Solvent_class = NA  
                    Solvent_support = NA  
                    Competitive_nucleophiles = NA  
                    Protecting_group_sensitivity = NA

                    5. Short explanatory paragraphs (3–10 sentences total)

                    After the above structured lines, you MUST write 3–10 sentences of coherent chemical reasoning, in free natural language, but still inside <think>.  

                    You MUST include:

                    - A short explanation of the reaction type: what transformation is occurring, and why you classified it as acid-amine amide coupling or “other”.  
                    - A description of the local structural environment around the reaction center, using the ring / FG / viewpoint information:
                    • ring systems (fused vs isolated, aromatic vs aliphatic),  
                    • nearby functional groups (carboxylic acid, amide, sulfonyl, CF3, alcohol, etc.),  
                    • steric effects (hindered acid / amine, bulky substituents),  
                    • electronic effects (EWG/EDG that change electrophilicity of the acyl center or nucleophilicity of the amine).  
                    - A brief discussion of the conditions:
                    • how well the reagents activate the acid (if at all),  
                    • whether the base is appropriate for deprotonation and proton shuttling,  
                    • whether the solvent class is favorable or unfavorable for amide formation,  
                    • whether competing nucleophiles or sensitive protecting groups are likely to interfere.  
                    - An overall mechanistic feasibility judgment: why, combining structure and conditions, the reaction is more likely to be high or low yield.

                    Keep this explanation section within about 10 sentences. Focus on chemistry and mechanism, not on data or training.

                    Example skeleton inside <think> (for guidance):

                    <think>
                    Reaction_type = acid-amine amide coupling
                    Reaction_site = [2, 4]

                    Acid_atom = 2
                    Amine_atom = 4
                    Acid_functional_group = carboxylic acid
                    Amine_functional_group = secondary aliphatic amine
                    Local_environment_key_groups = acid_fg: carboxylic acid [2,3,39]; nucleophile_fg: secondary aliphatic amine [4]; neighbors: secondary amide [20,21,22], sulfonyl [9,10,11], tertiary alcohol [28], CF3 [29-36]

                    Acid_activation = present (polyphosphate/T3P-like coupling reagent in solvent field)
                    Base_situation = suitable_base_present (triethylamine in solvent; tertiary aliphatic amine base)
                    Solvent_class = nonpolar (ClCCl, dichloromethane-like)
                    Solvent_support = supportive (nonpolar aprotic medium compatible with T3P and amide coupling)
                    Competitive_nucleophiles = [22, 28] (amide NH and tertiary OH present but relatively weak/sterically hindered nucleophiles)
                    Protecting_group_sensitivity = compatible (silyl and CF3 groups stable under mild base and coupling conditions)

                    The reaction transforms a simple carboxylic acid side chain into an amide by coupling with a secondary aliphatic amine embedded in a fused bicyclic system, forming a new C(=O)–N bond. The acid carbonyl at atom 2 is relatively unhindered and electronically neutral, while the amine at atom 4 is the most nucleophilic nitrogen in the molecule, not deactivated by resonance. Nearby sulfonyl and amide groups are electron-withdrawing, which slightly increases the acidity of the carboxylic acid and can also rigidify the scaffold but do not directly compete as nucleophiles. The conditions include a polyphosphate/T3P-type coupling reagent that can activate the carboxylic acid to a reactive mixed anhydride or similar intermediate, and a tertiary amine base that manages proton transfers without being nucleophilic. The nonpolar aprotic solvent is well-suited for this type of coupling and is compatible with the coupling reagent. Although there are other heteroatoms in the structure (amide NH, tertiary OH), their nucleophilicity is reduced by resonance or steric hindrance, so the main amine remains the preferred nucleophile. Overall, the combination of a good nucleophile, a reasonably accessible acid, appropriate activation chemistry, and compatible solvent/base strongly supports a high-yielding acid–amine amide coupling.
                    </think>

                    (Your actual content will differ; the above is only an example of format and level of detail.)

                    ------------------------------------------------
                    <answer> REQUIREMENTS

                    After </think>, you MUST output exactly one of:

                    <answer>high</answer>
                    or
                    <answer>low</answer>

                    - “high” means the reaction is more likely to have yield >20%.  
                    - “low” means the reaction is more likely to have yield ≤20%.  

                    Do NOT include any other text in <answer>.  
                    Do NOT mention numerical yields, optimization, data, training, probability, or uncertainty.""")


# Acidamine GPT-5 annotation prompt: answer only
def get_acidamine_only_answer_prompt(reaction_data, reactants_description, gt):
    return dedent(f"""\
                    You are a senior organic chemist specializing in reaction mechanisms, reactivity prediction, and structure–reactivity relationships.

                    CONTEXT  
                    - You are given:  
                    • Raw reaction data (including cano_rxn_smiles / mapped_rxn_smiles, catalysts, reagents, solvent).  
                    • A detailed reaction-center annotation (ring systems, functional groups, connectivity, viewpoints, atom mapping).  

                    YOUR TASK  
                    Analyze the given reaction, identify its reaction type, and decide whether, under the given conditions, the reaction is more likely to have a high or low yield. Yield >20% is considered “high”.

                    You MUST reason explicitly about:  
                    - The local structural environment around the reaction center (rings, functional groups, sterics, electronics).  
                    - The suitability of the conditions (catalysts / reagents / solvent) for the identified reaction type.  
                    - For acid–amine amide coupling specifically (“acidamine”), whether the most important factors are present, missing, or conflicting in the reaction entry.

                    ------------------------------------------------
                        
                    INPUT

                    REACTION DATA:
                    {reaction_data}

                    REACTION-CENTER ANNOTATION:
                    {reactants_description}

                    OUTPUT FORMAT (VERY IMPORTANT)

                    You MUST output:
                    <answer>high or low</answer>

                    No other text before or after.""")

# Acidamine model training prompt
def get_acidamine_simple_prompt(reaction_data):
    return dedent(f"""\
                    You are a senior organic chemist specializing in reaction mechanisms, reactivity prediction, and structure–reactivity relationships.

                    CONTEXT  
                    - You are given:  
                    • Raw reaction data (including cano_rxn_smiles / mapped_rxn_smiles, catalysts, reagents, solvent).  
                    • A detailed reaction-center annotation (ring systems, functional groups, connectivity, viewpoints, atom mapping).  

                    YOUR TASK  
                    Analyze the given reaction, identify its reaction type, and decide whether, under the given conditions, the reaction is more likely to have a high or low yield. Yield >20% is considered “high”.

                    You MUST reason explicitly about:  
                    - The local structural environment around the reaction center (rings, functional groups, sterics, electronics).  
                    - The suitability of the conditions (catalysts / reagents / solvent) for the identified reaction type.  
                    - For acid-amine amide coupling specifically (“acidamine”), whether the most important factors are present, missing, or conflicting in the reaction entry.

                    ------------------------------------------------
                        
                    INPUT

                    Reaction data:{reaction_data}
                    ------------------------------------------------
                    OUTPUT FORMAT (VERY IMPORTANT)

                    You MUST output:

                    <think>
                    ...structured fields...
                    ...short explanation sentences...
                    </think>
                    <answer>high or low</answer>

                    No other text before or after.""")

# Suzuki GPT-5 annotation prompt
def get_suzuki_prompt(reaction_data, reactants_description, gt):
    return dedent(f"""\
                    You are a senior organic chemist specializing in reaction mechanisms, reactivity prediction, and structure–reactivity relationships.
                  
                    You are given:
                    • Raw reaction data (cano_rxn_smiles / mapped_rxn_smiles, catalysts, reagents, solvent).
                    • A detailed reaction-center annotation (ring systems, functional groups, connectivity, viewpoints, atom mapping).

                    YOUR TASK
                    Analyze the given reaction, identify its reaction type, and decide whether, under the given conditions, the reaction is more likely to have a high or low yield. Yield >20% is considered “high”.

                    CRITICAL FOCUS (Suzuki yield drivers)
                    When the reaction is Suzuki-Miyaura, your reasoning MUST prioritize (in this order):
                    1) Electrophile class & activation difficulty (Ar–I/Br/Cl/OTf; heteroaryl; vinyl; alkyl; sterics at/near C–X).
                    2) Nucleophile class & stability (boronic acid/ester/trifluoroborate; heteroaryl boron; protodeboronation risk).
                    3) Catalyst system match to substrate difficulty (Pd source + ligand class; ability for Ar–Cl / hindered / heteroaryl).
                    4) Base suitability for transmetalation (present? strength? compatibility).
                    5) Solvent system enabling boron activation / phase transfer (aqueous mixed vs polar aprotic vs protic).
                    6) Catalyst poisons / strong coordination near the site (N/S/P donors).
                    7) Dominant side reactions (protodeboronation, homocoupling, dehalogenation, hydrolysis, β-hydride issues).

                    You MUST base all claims ONLY on the provided reaction entry and annotations. If something is not explicitly present, mark it as unclear/NA (do not assume temperature, additives, atmosphere, or water unless the solvent list implies it).

                    ------------------------------------------------
                        
                    INPUT

                    REACTION DATA:
                    {reaction_data}

                    REACTION-CENTER ANNOTATION:
                    {reactants_description}
                    ------------------------------------------------

                    I will privately tell you that the yield level of the following reaction is {gt}. Please act as if you do not know this information.
                    
                    ------------------------------------------------  
                  
                  
                    OUTPUT FORMAT (VERY IMPORTANT)
                    You MUST output:
                    <think> ...structured fields... ...short explanation sentences... </think> <answer>high or low</answer>
                    No other text before or after.

                    DETAILED REQUIREMENTS FOR <think>

                    (1) Reaction type identification (MANDATORY)
                    Include exactly one line:
                    Reaction_type = ...
                    If Suzuki, write exactly:
                    Reaction_type = Suzuki-Miyaura cross-coupling
                    Otherwise:
                    Reaction_type = other

                    (2) Reaction-center identification (MANDATORY)
                    Include exactly one line:
                    Reaction_site = [a, b]
                    Where a and b are the atom indices (from the mapped SMILES) of the two carbon atoms that form the NEW C–C bond.

                    (3) Suzuki-specific structural fields
                    If Reaction_type = Suzuki-Miyaura cross-coupling, you MUST provide ALL lines below:
                    Electrophile_atom = i
                    Nucleophile_atom = j
                    Electrophile_identity = ...
                    Nucleophile_identity = ...
                    Local_environment_key_features = ...
                    Guidance:
                    - Electrophile_identity must specify leaving group class (aryl iodide/bromide/chloride/triflate/vinyl halide/heteroaryl halide, etc.).
                    - Nucleophile_identity must specify boron class (boronic acid/boronate ester/trifluoroborate, etc.).
                    - Local_environment_key_features MUST list the closest steric/electronic/coordination features (with atom indices) that materially affect oxidative addition, transmetalation, or reductive elimination:
                    • Ortho substitution near electrophile carbon (steric hindrance)
                    • Strong EWGs/EDGs affecting oxidative addition/reductive elimination
                    • Nearby heteroatoms (N/S/P/O) that can coordinate Pd
                    • Functional groups that are base-sensitive or can bind Pd

                    If Reaction_type ≠ Suzuki-Miyaura cross-coupling, still include the same lines but set each to NA.

                    (4) Suzuki condition checklist (ONLY if Suzuki; else set all to NA)
                    Catalyst_present_and_suitable = STATUS (reason...)
                    Base_present_and_suitable = STATUS (reason...)
                    Solvent_class_and_suitability = STATUS (reason...)
                    Electrophile_reactivity = STATUS (reason...)
                    Potential_catalyst_poisons = [...] (short note)
                    Potential_side_reactions = STATUS (reason...)

                    Allowed STATUS values:
                    Catalyst_present_and_suitable:
                    suitable_present / unsuitable_present / absent / unclear
                    Base_present_and_suitable:
                    suitable_base_present / base_missing / base_too_weak / base_too_strong / unclear
                    Solvent_class_and_suitability:
                    aqueous_mixed_supportive / polar_aprotic_supportive / medium_polarity_supportive / protic_hindering / unclear
                    Electrophile_reactivity:
                    high (Ar-I/OTf) / moderate (Ar-Br) / low (Ar-Cl) / very_low (Ar-F) / unclear
                    Potential_side_reactions:
                    likely / unlikely / unclear

                    (5) Top yield limiters (MANDATORY; ONLY if Suzuki, else NA)
                    You MUST output exactly one line:
                    Top_yield_limiters = [CODE1; CODE2; CODE3]

                    Rules:
                    - Choose exactly 3 codes from the ENUM below (no free text).
                    - If fewer than 3 clearly apply, use M00 (insufficient_information) to fill remaining slots.
                    - Codes MUST be ordered by impact (most decisive first).
                    - Use ONLY what is explicitly supported by the reaction data + annotation.

                    ENUM (Top_yield_limiters codes)
                    M00 = insufficient_information (key catalyst/base/partners ambiguous or missing)
                    C01 = catalyst_absent_or_not_listed
                    C02 = catalyst_unclear_identity_or_loading (cannot assess suitability)
                    C03 = catalyst_likely_unsuitable_for_aryl_chloride_or_hindered_substrate (weak/unspecified ligand)
                    C04 = catalyst_poison_present_strong_S_donor (thiol/thioether/thiophene near site)
                    C05 = catalyst_poison_present_strong_N_donor (pyridine/imidazole/amine near site)
                    C06 = catalyst_poison_present_strong_P_donor_or_other_strong_ligand_competitor
                    B01 = base_missing_or_not_listed
                    B02 = base_too_weak_for_transmetalation (e.g., no effective inorganic base implied)
                    B03 = base_too_strong_or_incompatible (likely substrate decomposition / side reactions)
                    S01 = solvent_protic_only_or_strongly_protic (suppresses / destabilizes catalytic cycle)
                    S02 = solvent_not_supportive_for_boron_activation (no aqueous/mixed or poor polarity implied)
                    S03 = solvent_phase_transfer_issue (biphasic without enabling components implied)
                    E01 = electrophile_low_reactivity_aryl_chloride
                    E02 = electrophile_very_low_reactivity_aryl_fluoride_or_inert_CX
                    E03 = electrophile_sterically_hindered_at_CX (ortho,ortho or crowded site)
                    E04 = electrophile_heteroaryl_halide_coordination_risk (heteroaryl near OA site)
                    N01 = nucleophile_prone_to_protodeboronation (heteroaryl boronic acid / sensitive boron type)
                    N02 = nucleophile_low_transmetalation_efficiency (e.g., boronate ester without activation implied)
                    N03 = nucleophile_decomposition_or_incompatibility_with_base (functional group conflicts)
                    X01 = side_reaction_likely_homocoupling
                    X02 = side_reaction_likely_dehalogenation
                    X03 = side_reaction_likely_hydrolysis_or_deprotection
                    X04 = alkyl_partner_beta_hydride_issue (if alkyl involved)

                    (6) Critical missing or conflicting essentials (MANDATORY; ONLY if Suzuki, else NA)
                    Include exactly one line:
                    Critical_missing_or_conflicting = ...
                    Rules:
                    - If catalyst or base is absent/unclear, say so.
                    - If solvent suggests protic-only with no base/catalyst clarity, flag it.
                    - If mapping/annotation cannot support the bond-forming pair, flag it.
                    - Keep to one short sentence.

                    (7) Short explanatory paragraphs (3–8 sentences total)
                    After the structured lines, write 3–8 sentences of chemical reasoning, tightly focused on:
                    - Why it is Suzuki (or not).
                    - Oxidative addition feasibility given electrophile identity + local sterics/electronics.
                    - Transmetalation feasibility given boron class + base + solvent class.
                    - Whether catalyst system matches substrate difficulty (especially for Ar–Cl / hindered / heteroaryl).
                    - Poisons and the most likely side reaction channel.
                    - Conclude with an overall mechanistic feasibility judgement tied to the Top_yield_limiters.

                    <answer> REQUIREMENTS
                    After </think>, output exactly one of:
                    <answer>high</answer>
                    or
                    <answer>low</answer>
                    Do NOT include any other text in <answer>.
                    Do NOT mention numerical yields, optimization, probability, or uncertainty.
                    """)


# Suzuki GPT-5 annotation prompt: answer only
def get_suzuki_only_answer_prompt(reaction_data, reactants_description, gt):
    return dedent(f"""\
                    You are a senior organic chemist specializing in reaction mechanisms, reactivity prediction, and structure–reactivity relationships.

                    You are given:
                    • Raw reaction data (cano_rxn_smiles / mapped_rxn_smiles, catalysts, reagents, solvent).
                    • A detailed reaction-center annotation (ring systems, functional groups, connectivity, viewpoints, atom mapping).

                    YOUR TASK
                    Analyze the given reaction, identify its reaction type, and decide whether, under the given conditions, the reaction is more likely to have a high or low yield. Yield >20% is considered “high”.

                    CRITICAL FOCUS (Suzuki yield drivers)
                    When the reaction is Suzuki-Miyaura, your reasoning MUST prioritize (in this order):
                    1) Electrophile class & activation difficulty (Ar–I/Br/Cl/OTf; heteroaryl; vinyl; alkyl; sterics at/near C–X).
                    2) Nucleophile class & stability (boronic acid/ester/trifluoroborate; heteroaryl boron; protodeboronation risk).
                    3) Catalyst system match to substrate difficulty (Pd source + ligand class; ability for Ar–Cl / hindered / heteroaryl).
                    4) Base suitability for transmetalation (present? strength? compatibility).
                    5) Solvent system enabling boron activation / phase transfer (aqueous mixed vs polar aprotic vs protic).
                    6) Catalyst poisons / strong coordination near the site (N/S/P donors).
                    7) Dominant side reactions (protodeboronation, homocoupling, dehalogenation, hydrolysis, β-hydride issues).

                    You MUST base all claims ONLY on the provided reaction entry and annotations. If something is not explicitly present, mark it as unclear/NA (do not assume temperature, additives, atmosphere, or water unless the solvent list implies it).

                    ------------------------------------------------
                        
                    INPUT

                    REACTION DATA:
                    {reaction_data}

                    REACTION-CENTER ANNOTATION:
                    {reactants_description}
                  
                    OUTPUT FORMAT (VERY IMPORTANT)
                    You MUST output:
                    <answer>high or low</answer>
                    No other text before or after.""")


# Suzuki model training prompt
def get_suzuki_simple_prompt(reaction_data):
    return dedent(f"""\
                You are a senior organic chemist specializing in reaction mechanisms, reactivity prediction, and structure–reactivity relationships.

                You are given:
                • Raw reaction data (cano_rxn_smiles / mapped_rxn_smiles, catalysts, reagents, solvent).
                • A detailed reaction-center annotation (ring systems, functional groups, connectivity, viewpoints, atom mapping).

                YOUR TASK
                Analyze the given reaction, identify its reaction type, and decide whether, under the given conditions, the reaction is more likely to have a high or low yield. Yield >20% is considered “high”.

                CRITICAL FOCUS (Suzuki yield drivers)
                When the reaction is Suzuki-Miyaura, your reasoning MUST prioritize (in this order):
                1) Electrophile class & activation difficulty (Ar–I/Br/Cl/OTf; heteroaryl; vinyl; alkyl; sterics at/near C–X).
                2) Nucleophile class & stability (boronic acid/ester/trifluoroborate; heteroaryl boron; protodeboronation risk).
                3) Catalyst system match to substrate difficulty (Pd source + ligand class; ability for Ar–Cl / hindered / heteroaryl).
                4) Base suitability for transmetalation (present? strength? compatibility).
                5) Solvent system enabling boron activation / phase transfer (aqueous mixed vs polar aprotic vs protic).
                6) Catalyst poisons / strong coordination near the site (N/S/P donors).
                7) Dominant side reactions (protodeboronation, homocoupling, dehalogenation, hydrolysis, β-hydride issues).

                You MUST base all claims ONLY on the provided reaction entry and annotations. If something is not explicitly present, mark it as unclear/NA (do not assume temperature, additives, atmosphere, or water unless the solvent list implies it).

                ------------------------------------------------
                    
                INPUT

                Reaction data: {reaction_data}
                ------------------------------------------------
                
                OUTPUT FORMAT (VERY IMPORTANT)
                You MUST output:
                <think> ...structured fields... ...short explanation sentences... </think> <answer>high or low</answer>
                No other text before or after.

                (1) Top yield limiters (MANDATORY; ONLY if Suzuki, else NA)
                You MUST output exactly one line:
                Top_yield_limiters = [CODE1; CODE2; CODE3]

                Rules:
                - Choose exactly 3 codes from the ENUM below (no free text).
                - If fewer than 3 clearly apply, use M00 (insufficient_information) to fill remaining slots.
                - Codes MUST be ordered by impact (most decisive first).
                - Use ONLY what is explicitly supported by the reaction data + annotation.

                ENUM (Top_yield_limiters codes)
                M00 = insufficient_information (key catalyst/base/partners ambiguous or missing)
                C01 = catalyst_absent_or_not_listed
                C02 = catalyst_unclear_identity_or_loading (cannot assess suitability)
                C03 = catalyst_likely_unsuitable_for_aryl_chloride_or_hindered_substrate (weak/unspecified ligand)
                C04 = catalyst_poison_present_strong_S_donor (thiol/thioether/thiophene near site)
                C05 = catalyst_poison_present_strong_N_donor (pyridine/imidazole/amine near site)
                C06 = catalyst_poison_present_strong_P_donor_or_other_strong_ligand_competitor
                B01 = base_missing_or_not_listed
                B02 = base_too_weak_for_transmetalation (e.g., no effective inorganic base implied)
                B03 = base_too_strong_or_incompatible (likely substrate decomposition / side reactions)
                S01 = solvent_protic_only_or_strongly_protic (suppresses / destabilizes catalytic cycle)
                S02 = solvent_not_supportive_for_boron_activation (no aqueous/mixed or poor polarity implied)
                S03 = solvent_phase_transfer_issue (biphasic without enabling components implied)
                E01 = electrophile_low_reactivity_aryl_chloride
                E02 = electrophile_very_low_reactivity_aryl_fluoride_or_inert_CX
                E03 = electrophile_sterically_hindered_at_CX (ortho,ortho or crowded site)
                E04 = electrophile_heteroaryl_halide_coordination_risk (heteroaryl near OA site)
                N01 = nucleophile_prone_to_protodeboronation (heteroaryl boronic acid / sensitive boron type)
                N02 = nucleophile_low_transmetalation_efficiency (e.g., boronate ester without activation implied)
                N03 = nucleophile_decomposition_or_incompatibility_with_base (functional group conflicts)
                X01 = side_reaction_likely_homocoupling
                X02 = side_reaction_likely_dehalogenation
                X03 = side_reaction_likely_hydrolysis_or_deprotection
                X04 = alkyl_partner_beta_hydride_issue (if alkyl involved)""")


def get_buchwald_simple_prompt(reaction_data):
    return dedent(f"""\
        You are a senior organic chemist specializing in reaction mechanisms, reactivity prediction, and structure–reactivity relationships.

        CONTEXT  
        - You are given:  
        • Raw reaction data (including cano_rxn_smiles / mapped_rxn_smiles, catalyst, ligand, base, additive, solvent).  
        • A detailed reaction-center annotation (ring systems, functional groups, connectivity, viewpoints, atom mapping).  

        YOUR TASK  
        Analyze the given reaction, identify its reaction type, and decide whether, under the given conditions, the reaction is more likely to have a high or low yield. Yield >20% is considered "high".

        You MUST reason explicitly about:  
        - The local structural environment around the reaction center (rings, functional groups, sterics, electronics).  
        - The suitability of the conditions (catalyst, ligand, base, additive, and solvent) for the identified reaction type.  
        - Whether key factors for the inferred reaction type (e.g., necessary activating groups, catalysts, bases, oxidants/reductants, solvent compatibility, temperature sensitivity) are present, missing, or conflicting in the entry.

        ------------------------------------------------

        INPUT

        Reaction data: {reaction_data}
        ------------------------------------------------
        OUTPUT FORMAT (VERY IMPORTANT)

        You MUST output:

        <think>
        ...structured fields...
        ...short explanation sentences...
        </think>
        <answer>high or low</answer>

        No other text before or after.""")


def get_buchwald_prompt(reaction_data, reactants_description, gt):
    return dedent(f"""\
                You are an expert organic chemist specializing in reaction mechanisms, reactivity prediction, and structure–reactivity relationships.

                Task:
                Given reaction data and reaction-center annotation, identify whether the reaction is Buchwald–Hartwig amination and predict whether the reaction yield is likely high or low. Yield >20% is considered high.

                Important rules:
                - Use ONLY the provided reaction entry and annotation.
                - Do NOT assume temperature, atmosphere, additives, water, or missing reagents.
                - If a key fact is not explicitly present, write unclear or NA.
                - For Buchwald analysis, prioritize:
                1) electrophile type/reactivity (Ar–I/Br/Cl/OTf, heteroaryl, sterics),
                2) amine type/reactivity (alkyl, cyclic, aniline, hindered, deactivated),
                3) Pd/ligand suitability,
                4) base suitability,
                5) solvent suitability,
                6) catalyst poisons / coordinating groups,
                7) likely side reactions or catalyst inhibition.

                INPUT
                REACTION DATA:
                {reaction_data}

                REACTION-CENTER ANNOTATION:
                {reactants_description}

                You are privately told the true yield label is {gt}, but you must behave as if you do not know it.

                Output format:
                <think>
                Reaction_type = ...
                Reaction_site = [a, b]
                Electrophile_atom = ...
                Nucleophile_atom = ...
                Electrophile_identity = ...
                Nucleophile_identity = ...
                Local_environment_key_features = ...
                Catalyst_present_and_suitable = ...
                Base_present_and_suitable = ...
                Solvent_class_and_suitability = ...
                Electrophile_reactivity = ...
                Amine_reactivity = ...
                Potential_catalyst_poisons = [...]
                Potential_side_reactions = ...
                Reasoning = ...
                </think>
                <answer>high or low</answer>

                Formatting rules:
                1) If the reaction is Buchwald–Hartwig amination, write exactly:
                Reaction_type = Buchwald–Hartwig amination
                Otherwise write:
                Reaction_type = other

                2) Reaction_site = [a, b]
                Where a and b are the two atom indices in mapped SMILES that form the new C–N bond.

                3) If Reaction_type = Buchwald–Hartwig amination:
                - Electrophile_atom = aryl/vinyl carbon bearing the leaving group
                - Nucleophile_atom = amine nitrogen forming the new bond
                - Electrophile_identity must state substrate class, e.g. aryl bromide / aryl chloride / heteroaryl chloride / aryl triflate
                - Nucleophile_identity must state amine class, e.g. aniline / primary alkyl amine / secondary cyclic amine / heteroarylamine
                - Local_environment_key_features should briefly mention nearest important steric/electronic/coordination effects with atom indices

                4) If Reaction_type = other:
                Set all remaining fields to NA.

                Allowed values:
                Catalyst_present_and_suitable = suitable_present / unsuitable_present / absent / unclear
                Base_present_and_suitable = suitable_base_present / base_missing / base_too_weak / base_too_strong / unclear
                Solvent_class_and_suitability = nonpolar_aromatic_supportive / ether_supportive / polar_aprotic_supportive / protic_hindering / unclear
                Electrophile_reactivity = high (Ar-I/OTf) / moderate (Ar-Br) / low (Ar-Cl) / very_low (Ar-F) / unclear
                Amine_reactivity = high (alkyl/cyclic amine) / moderate (aniline) / low (electron-poor or hindered aniline) / very_low (amide-like or strongly deactivated N) / unclear
                Potential_side_reactions = likely / unlikely / unclear

                Reasoning requirements:
                - Keep it short: 2–4 sentences only.
                - Must mention electrophile difficulty, amine reactivity, catalyst/base/solvent match, and major poison/side-reaction risk if present.
                - Final judgment should support the high/low answer.""")
