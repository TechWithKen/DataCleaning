import os
import pandas as pd
import re
import numpy as np


def get_filename(path):
    filename = os.path.basename(path)
    name = re.split(r'[_\.]', filename)[0]

    key = name[:3]
    return key



def get_r_re_filename(path):
    name = os.path.basename(path)
    name = name.rsplit(".", 1)[0]         
    name = name.replace("_RE", "").replace("_R", "")
    return name



def get_peptide_isr(folder_path):
    c_peptide_values = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith("txt") and not f.lower().endswith(("_re.txt", "_r.txt",))]
    isr_values = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(("_re.txt", "_r.txt"))]


    c_peptide_database = {}
    isr_values_database = {}

    for file in c_peptide_values:

        key = get_filename(file)

        if key not in c_peptide_database:
            c_peptide_database[key] = []


        c_peptide_database[key].append(file)

    for file in isr_values:
        key = get_filename(file)


        if key not in isr_values_database:
            isr_values_database[key] = []


        isr_values_database[key].append(file)

    return {"C_peptide": c_peptide_database, "ISR": isr_values_database}

path_to_folder = "/Users/alert/Desktop/ISEC/ISR-PEPTIDE-TXT/"



def preprocess_file(filename, col_name=None):
    with open(filename, "r") as f:
        lines = f.readlines()
    
    # Extract metadata
    metadata = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    
    df_meta = pd.DataFrame(
        metadata.items(),
        columns=["TIME(min)", "C_PEPTIDE"]
    )

    table_start = 0
    for i, line in enumerate(lines):
        if "TIME (min)" in line and "C-PEPTIDE" in line:
            table_start = i + 1 
            break


    df = pd.read_csv(
        filename,
        sep=r"\s+",
        skiprows=table_start,
        engine="python",
        names=["TIME(min)", "C_PEPTIDE"],
        on_bad_lines='skip'
    )


    new_dataframe  = pd.concat([df, df_meta])
    datasetframe = new_dataframe.reset_index().drop(columns=["index"])

    # Optonally rename the column to avoid conflicts when merging
    if col_name:
        datasetframe.rename(columns={"C_PEPTIDE": col_name}, inplace=True)

    return datasetframe, metadata["ID"]


c_pep = get_peptide_isr(path_to_folder)["C_peptide"]
c_pep_dataframes = {}
for files in c_pep:
    
    c_pep_final_df = preprocess_file(c_pep[files][0], col_name=f"{preprocess_file(c_pep[files][0])[1]}")[0]
    

    # Step 2: Loop through the rest and merge
    for idx, file in enumerate(c_pep[files][1:], start=2):
        id = preprocess_file(file)[1]
        
        key = get_filename(file)
        df_new = preprocess_file(file, col_name=f"{id}.")[0]
        c_pep_final_df = pd.merge(c_pep_final_df, df_new, on="TIME(min)", how="outer")

    key = get_filename(files)

    if key not in c_pep_dataframes:
        c_pep_dataframes[key] = c_pep_final_df


nap = c_pep_dataframes["NAP"]
nap.rename(columns=lambda c: f"N{c}" if c.startswith("APR") else c, inplace=True)




### ISR File Manipulation#####
isr = get_peptide_isr(path_to_folder)["ISR"]

file_database = isr

def preprocess_isr(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    table_start = 0
    for i, line in enumerate(lines):
        if "from" in line and "to" in line:
            table_start = i + 1
            break

  
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=table_start,
        engine="python",
        names=["from", "to", "isr"],
        on_bad_lines="skip"
    )


    df.loc[df.index[0], "isr"] = df.loc[df.index[0], "to"]
    df.loc[df.index[0], "to"]  = df.loc[df.index[0], "from"]


    mask = df.isna().any(axis=1)
    first_none_idx = mask.idxmax() if mask.any() else len(df)
    df = df.iloc[:first_none_idx]

    subject = get_r_re_filename(filepath).upper()

    df = (
        df.drop(columns=["from"])
          .rename(columns={
              "to": "TIME(min)",
              "isr": subject
          })
    )

    return df


def create_isr_dataframe():
    isr_dataframe = {}

    for group, group_files in file_database.items():

        isr_final_df = None

        for file in group_files:
            df = preprocess_isr(file)

            if isr_final_df is None:
                isr_final_df = df
            else:
                isr_final_df = pd.merge(
                    isr_final_df,
                    df,
                    on="TIME(min)",
                    how="outer"   # IMPORTANT
                )
            key = get_filename(file)

        isr_final_df = isr_final_df.sort_values("TIME(min)").reset_index(drop=True)
        

        if key not in isr_dataframe:
            isr_dataframe[key] = isr_final_df
            
    return isr_dataframe

def expand_isr(dataframe_to_expand):
    insulin_df = dataframe_to_expand.melt(
    id_vars=["TIME(min)"],
    var_name="sample",
    value_name="ISR",
)
    insulin_df["sample_id"] = insulin_df["sample"].astype("category").cat.codes + 1

    insulin_df = insulin_df.sort_values(by=["sample_id", "TIME(min)"], ascending=[True, True]).reset_index(drop=True)

    insulin_df
    return insulin_df



# print(create_isr_dataframe()["JAN"])
def expand_dataframe(dataframe_to_expand):


    each_month_dataframe = dataframe_to_expand.copy()
    age_idx = each_month_dataframe.index[each_month_dataframe["TIME(min)"].isin(["AGE", "ID"])][0]
    top_dataframe = each_month_dataframe.iloc[:age_idx]
    bottom_dataframe = each_month_dataframe.iloc[age_idx:]

    expand_top = top_dataframe.melt(
        id_vars=["TIME(min)"],
        var_name="sample",
        value_name="value",
    )
    expand_top["sample_id"] = expand_top["sample"].astype("category").cat.codes + 1

    expand_top = expand_top.sort_values(by=["sample_id", "TIME(min)"], ascending=[True, True]).reset_index(drop=True)



    expand_bottom = bottom_dataframe

    expand_bottom = expand_bottom.set_index("TIME(min)")
    expand_bottom = expand_bottom.T


    expand_bottom = expand_bottom.reset_index()
    numeric_cols = ["AGE", "HEIGHT", "ID", "SEX", "SUBJECT", "WEIGHT"]
    expand_bottom.rename(columns={"index":"Sample_ID"}, inplace=True)


    expand_top = expand_top.merge(
        expand_bottom[["Sample_ID", "AGE", "HEIGHT", "ID", "SEX", "SUBJECT", "WEIGHT"]],
        left_on="sample", 
        right_on="Sample_ID",
        how="left"
    )


    expand_top = (
        expand_top.drop(columns=["sample"]).rename(columns={"value": "C-Peptide"})
    )

    expand_top["C-Peptide"] = pd.to_numeric(expand_top["C-Peptide"], errors="coerce")
    return expand_top



ultimate_dataframe = pd.DataFrame()
current_max_sample_id = 0

for pick_month in c_pep_dataframes:

    if pick_month == "nap": 
        final_dataframe = nap
    final_dataframe = expand_dataframe(c_pep_dataframes[pick_month.upper()])
    insulin = expand_isr(create_isr_dataframe()[pick_month.upper()])


    final_dataframe["ID"] = final_dataframe["Sample_ID"]
    final_dataframe.drop(columns=["Sample_ID"], inplace=True)
    final_dataframe["ID"] = (
        final_dataframe["ID"]
        .str.translate(str.maketrans({" ": "", ".": ""}))
    )


    final_dataframe = pd.merge(final_dataframe, insulin, left_on=["TIME(min)", "ID"], right_on=["TIME(min)", "sample"], how="inner")
    final_dataframe.rename(columns={"sample_id_x":"Sample ID"}, inplace=True)
    final_dataframe.drop(columns=["ID", "sample_id_y"], inplace=True)

    # safe cumulative addition
    final_dataframe["Sample ID"] = final_dataframe["Sample ID"].astype("int64") + (current_max_sample_id)

    # update max Sample ID safely
    current_max_sample_id = final_dataframe["Sample ID"].max()

    ultimate_dataframe = pd.concat([ultimate_dataframe, final_dataframe], ignore_index=True)

ultimate_dataframe.to_excel("fulldataset.xlsx", index=False)