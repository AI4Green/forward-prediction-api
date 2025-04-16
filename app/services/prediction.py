import base64
import re
from io import BytesIO
from typing import Any, Dict, List

from fastapi.logger import logger
from rdkit import Chem

from app.schemas.prediction import Prediction as PredictionSchema
from app.schemas.prediction import PredictionResult
from app.services.opennmt_model_runner import OpenNMTModelRunner
from app.services.pubchem_api import PubchemAPI
from app.utils.reaction_drawer import draw_labeled_reaction_image


class Prediction:
    def __init__(self, model: OpenNMTModelRunner):
        self.model = model
        self.n_best = 30
        self.pubchem_api = PubchemAPI()

    def predict(self, smiles_list: List[str]) -> PredictionResult:
        """
        Takes a list of SMILES strings and returns the first N predictions (unsorted).
        """
        logger.info(f"Starting prediction for SMILES: {smiles_list}")

        tokenised = self._preprocess(smiles_list)
        results = self._inference(tokenised)

        first_n = self._get_top_n_predictions(results, n=5)[0]

        # Extract products and scores
        products = [entry["smiles"] for entry in first_n]
        scores = [entry["score"] for entry in first_n]

        reactants = self._parse_smiles(".".join(smiles_list))

        data: List[PredictionSchema] = []

        for product, score in zip(products, scores):
            mol = Chem.MolFromSmiles(product)
            name = self.pubchem_api.get_name_from_pubchem(product)
            iupac = name.iupac_name
            synonyms = name.synonyms
            label = iupac if iupac else product
            img = draw_labeled_reaction_image(reactants, mol, label)
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            data.append(
                PredictionSchema(
                    product=product,
                    score=score,
                    reaction_image=image_b64,
                    iupac_name=iupac,
                    synonyms=synonyms,
                )
            )

        return PredictionResult(result=data)

    def _get_top_n_predictions(
        self, results: List[Dict[str, Any]], n: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Takes raw model results and returns the first N predictions (unsorted).
        """
        top_n_outputs = []

        for result in results:
            products = result["products"]
            scores = result["scores"]

            combined = list(zip(products, scores))[:n]  # No sorting, just take first n
            top_n = [{"smiles": p, "score": round(s, 2)} for p, s in combined]
            top_n_outputs.append(top_n)

        return top_n_outputs

    def _preprocess(self, smiles_list: List[str]) -> List[Dict[str, str]]:
        """
        Preprocess a list of SMILES strings into tokenized input for the model.
        """
        tokenized_smiles = [
            {"src": self._smi_tokenizer(self._canonicalize_smiles(smi))}
            for smi in smiles_list
        ]
        return tokenized_smiles

    def _inference(self, tokenized_data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Run inference on tokenized SMILES using the model.
        """

        results = []

        try:
            products, scores, _, _, _ = self.model.run(tokenized_data)
            assert (
                len(tokenized_data) * self.n_best == len(products) == len(scores)
            ), "Output size doesn't match expected n_best"

            for i in range(len(tokenized_data)):
                prod_subset = products[i * self.n_best : (i + 1) * self.n_best]
                score_subset = scores[i * self.n_best : (i + 1) * self.n_best]

                valid_products = []
                valid_scores = []
                for prod, score in zip(prod_subset, score_subset):
                    product = "".join(prod.split())
                    if not self._canonicalize_smiles(product):
                        continue
                    valid_products.append(product)
                    valid_scores.append(score)

                results.append({"products": valid_products, "scores": valid_scores})

        except Exception as e:
            logger.error(f"model.run() failed in batch mode: {str(e)}")
            for item in tokenized_data:
                try:
                    products, scores, *_ = self.model.run(inputs=[item])
                except Exception as inner_e:
                    logger.error(f"model.run() failed on single input: {str(inner_e)}")
                    products, scores = [], []

                valid_products = []
                valid_scores = []
                for prod, score in zip(products, scores):
                    product = "".join(prod.split())
                    if not self._canonicalize_smiles(product):
                        continue
                    valid_products.append(product)
                    valid_scores.append(score)

                results.append({"products": valid_products, "scores": valid_scores})

        return results

    def _smi_tokenizer(self, smi: str):
        """Tokenize a SMILES molecule or reaction, adapted from https://github.com/pschwllr/MolecularTransformer"""
        pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        assert smi == "".join(tokens)

        return " ".join(tokens)

    def _canonicalize_smiles(self, smiles: str, remove_atom_number: bool = True):
        """Adapted from Molecular Transformer"""
        smiles = "".join(smiles.split())
        cano_smiles = ""

        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            if remove_atom_number:
                [a.ClearProp("molAtomMapNumber") for a in mol.GetAtoms()]

            cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            # Sometimes stereochem takes another canonicalization... (just in case)
            mol = Chem.MolFromSmiles(cano_smiles)
            if mol is not None:
                cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

        return cano_smiles

    def _parse_smiles(self, smi_str: str):
        """Parses a SMILES string and returns a list of RDKit Mol objects"""
        return [Chem.MolFromSmiles(smi) for smi in smi_str.split(".") if smi]
