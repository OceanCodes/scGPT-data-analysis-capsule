import anndata
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import joblib
import re
from scipy.stats import ttest_1samp
import itertools

mem = joblib.Memory("/scratch/cache", compress=5, verbose=0)


# Compile regex once and reuse
regex = re.compile(r"perturbation_experiment_([^_]+)_level")

# Constants
DATA_DIR = "/data/gene_perturbation_colon_epithelial_scGPT_with_binning_1"
FILE_PATTERN = DATA_DIR + "/perturbation_experiment_{gene}_level_{level}"
BASE_GENE = "JUND"
BASE_LEVEL = "1.0"
MODEL_PATH = "/code/colon_crohn_model_xgboost.joblib"
ADDITIONAL_DATA_DIRS = [
    "/data/perturbation_bining_colon_epothelial_take_7",
    "/data/colon_epithelial_cellxgene_perturbation_4/perturbations",
]
GENE_EXPRESSION_DATA = (
    "/data/colon_epithelial_cellxgene/ae802158-1a7e-43e0-9c23-cdc688ce3481.h5ad"
)
LEVELS_MAP = {0: "KO", 5.0: "OE", 1: "WT"}


@mem.cache
def load_base_data():
    base_fname = FILE_PATTERN.format(gene=BASE_GENE, level=BASE_LEVEL)
    base_embeddings = np.load(f"{base_fname}.embeddings.npy")
    adata = anndata.read_h5ad(f"{base_fname}.h5ad")
    base_df = adata.obs[
        ["tissue", "cell_type", "sex", "development_stage", "disease"]
    ].copy()
    colon_model = joblib.load(MODEL_PATH)
    colon_index = adata.obs.reset_index().query("tissue == 'colon'").index
    colon_embeddings = base_embeddings[colon_index]
    return adata, colon_model, colon_index, colon_embeddings


@mem.cache
def get_genes_from_dirs(data_dirs):
    genes = set()
    for data_dir in data_dirs:
        genes.update(
            regex.search(f.name).group(1) for f in list(Path(data_dir).glob("*.npz"))
        )
    return list(genes)


@mem.cache
def load_gene_expression_data(genes):
    gene_epression_adata = anndata.read_h5ad(GENE_EXPRESSION_DATA)
    genes_index = (
        gene_epression_adata.var.reset_index()
        .query("feature_name in @genes")[["feature_name"]]
        .reset_index()
    )
    df = pd.DataFrame(
        gene_epression_adata.X[:, genes_index["index"].tolist()].todense(),
        columns=genes_index["feature_name"].tolist(),
        index=gene_epression_adata.obs.reset_index()
        .set_index(["cell_id", "tissue", "cell_type", "development_stage", "disease"])
        .index,
    ).query("tissue == 'colon'")
    return df


@mem.cache
def calculate_relevant_genes(df):
    non_zero_fraction = df.apply(lambda x: (x > 0).sum() / len(x)).reset_index()
    non_zero_fraction["total_non_zero"] = df.apply(lambda x: (x > 0).sum()).values
    return non_zero_fraction.query("total_non_zero > 1000")["index"].tolist()


@mem.cache
def calculate_disease_proba_of_gene(params, df, colon_model, colon_index, data_dirs):
    gene, level = params
    try:
        df_ = (
            df[[gene, "disease_probability"]]
            .rename(columns={gene: "expression"})
            .copy()
        )
        df_["gene"] = gene
        df_["level"] = LEVELS_MAP[level]
        if level == 1:
            df_["pert_probability"] = df_["disease_probability"]
            df_["probability_change"] = 0
            return df_
        path = None
        for data_dir in data_dirs:
            candidate_path = f"{data_dir}/perturbation_experiment_{gene}_level_{level}.embeddings.npz"
            if Path(candidate_path).exists():
                path = candidate_path
                break
        if path:
            embeddings = np.load(path)["emb"][colon_index]
            df_["pert_probability"] = colon_model.predict_proba(embeddings)[:, 0]
            df_["probability_change"] = (
                df_["pert_probability"] - df_["disease_probability"]
            )
        print(gene, df_.shape)
        return df_
    except Exception as e:
        print(f"Error! {gene}: {e}")
        return pd.DataFrame([])


def t_test(group):
    _, p_value = ttest_1samp(group["probability_change"], 0)
    return p_value

@mem.cache
def add_stats(data):
    grouped = data.groupby(["gene", "level"])

    # Function to perform t-test
    results = grouped.apply(t_test).reset_index()
    results.columns = ["gene", "level", "p_value"]

    # Correct for multiple testing
    results["significance"] = results["p_value"].apply(
        lambda x: (
            "***" if x < 0.001 else "**" if x < 0.01 else "*" if x < 0.05 else "ns"
        )
    )

    stats = (
        grouped["probability_change"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .assign(stderr=lambda x: x["std"] / np.sqrt(x["count"]))
    )
    return results.merge(stats, how="left", on=["gene", "level"])



def main():
    adata, colon_model, colon_index, colon_embeddings = load_base_data()
    df = load_gene_expression_data(get_genes_from_dirs(ADDITIONAL_DATA_DIRS))
    df["disease_probability"] = colon_model.predict_proba(colon_embeddings)[:, 0]
    relevant_genes = calculate_relevant_genes(df)
    params = [(gene, level) for gene in relevant_genes for level in [0, 5.0, 1]]

    dfs = []
    for param in tqdm(params):
        proba = calculate_disease_proba_of_gene(
            param, df, colon_model, colon_index, ADDITIONAL_DATA_DIRS
        )
        if proba.shape[1] == 6:
            dfs.append(proba)
    dfs = pd.concat(dfs)
    data = dfs.query("expression > 0").query("abs(probability_change) > 0.05")
    results = add_stats(data)
    return results


if __name__ == "__main__":
    results = main()
    results.to_csv("/results/colon_epithelial_crohn_gene_perturbation.results.csv", index=False)

