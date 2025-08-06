import numpy as np
import csv
import os
from dataclasses import dataclass
from .add_tag import add_tag
from .scoring_utils import QED, LogP, SA_score, shannon_entropy_morgan, scaffold_uniqueness
from reinvent_plugins.components.component_results import ComponentResults

@add_tag("__component")
class Diversity:
    def __init__(self, params):
        self.params = params
        self.weight = [1.0]  # Can be adjusted or parameterized

    def __call__(self, smiles, valid_mask, **kwargs):
        # Apply valid mask
        valid_smiles = [s for s, v in zip(smiles, valid_mask) if v]
        n_total = len(smiles)
        rewards = np.zeros(n_total, dtype=np.float32)

        if not valid_smiles:
            return self._empty_result(n_total)

        # Compute raw metrics
        qed_scores = QED(valid_smiles)
        logp_scores = LogP(valid_smiles)
        sa_scores = SA_score(valid_smiles)
        binding_affinities = [0.0] * len(valid_smiles)
        random_seed = 0.0

        entropy = shannon_entropy_morgan(valid_smiles)
        scaff_unq = scaffold_uniqueness(valid_smiles)

        # Duplicated per-SMILES for table
        shannon_entropy = [entropy] * len(valid_smiles)
        scaffolds = [scaff_unq] * len(valid_smiles)

        # Compute reward
        normalized_entropy = entropy / np.log(2048)
        score_0_to_1 = 0.5 * normalized_entropy + 0.5 * scaff_unq
        score_scaled = float(score_0_to_1 * 10)
        reward_vals = [score_scaled] * len(valid_smiles)

        # Write CSV
        data = []
        for i, (smi, affinity, qed_val, logp_val, sa_val, se, scaff, reward) in enumerate(
            zip(valid_smiles, binding_affinities, qed_scores, logp_scores, sa_scores, shannon_entropy, scaffolds, reward_vals)
        ):
            row = [
                smi, affinity, qed_val, logp_val, sa_val, se, scaff, random_seed,
                0, 0, 0, 0, reward  # scaled metrics not used in this version
            ]
            data.append(row)

        file_exists = os.path.isfile("rl_generated_scaffolds_diversity_penalty_only_shannon_and_scaffold.csv")
        with open("rl_generated_scaffolds_diversity_penalty_only_shannon_and_scaffold.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow([
                    "SMILES", "Binding Affinity", "QED", "LogP", "SA Score",
                    "Shannon Entropy", "Scaffold uniqueness", "QuickVina Random Seed",
                    "QED Scaled", "LogP Scaled", "SA Score Scaled", "Affinity Scaled", "Final Reward"
                ])
            writer.writerows(data)

        # Insert reward back into full array
        rewards[valid_mask] = reward_vals

        metadata = {
            "component_type": "diversity",
            "transformed_scores": [rewards],
            "raw_scores": [rewards],
            "weight": self.weight
        }

        return ComponentResults(scores=[rewards])

    def _empty_result(self, n):
        metadata = {
            "component_type": "diversity",
            "transformed_scores": [np.zeros(n)],
            "raw_scores": [np.zeros(n)],
            "weight": self.weight
        }

        return ComponentResults(scores=[np.zeros(n)])