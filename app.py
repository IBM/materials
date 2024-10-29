import gradio as gr
from huggingface_hub import InferenceClient
import matplotlib.pyplot as plt
from PIL import Image
from rdkit.Chem import Descriptors, QED, Draw
from rdkit.Chem.Crippen import MolLogP
import pandas as pd
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem import DataStructs, AllChem
from transformers import BartForConditionalGeneration, AutoTokenizer, AutoModel
from transformers.modeling_outputs import BaseModelOutput
import selfies as sf
from rdkit import Chem
import torch
import numpy as np
import umap
import pickle
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
import json

import os

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

# my_theme = gr.Theme.from_hub("ysharma/steampunk")
# my_theme = gr.themes.Glass()

"""
# カスタムテーマ設定
theme = gr.themes.Default().set(
    body_background_fill="#000000",  # 背景色を黒に設定
    text_color="#FFFFFF",            # テキスト色を白に設定
)
"""
"""
import sys
sys.path.append("models")
sys.path.append("../models")
sys.path.append("../")"""


# Get the current file's directory
base_dir = os.path.dirname(__file__)
print("Base Dir : ", base_dir)

import models.fm4m as fm4m


# Function to display molecule image from SMILES
def smiles_to_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol)
        return img
    return None


# Function to get canonical SMILES
def get_canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    return None


# Dictionary for SMILES strings and corresponding images (you can replace with your actual image paths)
smiles_image_mapping = {
    "Mol 1": {"smiles": "C=C(C)CC(=O)NC[C@H](CO)NC(=O)C=Cc1ccc(C)c(Cl)c1", "image": "img/img1.png"},
    # Example SMILES for ethanol
    "Mol 2": {"smiles": "C=CC1(CC(=O)NC[C@@H](CCCC)NC(=O)c2cc(Cl)cc(Br)c2)CC1", "image": "img/img2.png"},
    # Example SMILES for butane
    "Mol 3": {"smiles": "C=C(C)C[C@H](NC(C)=O)C(=O)N1CC[C@H](NC(=O)[C@H]2C[C@@]2(C)Br)C(C)(C)C1",
              "image": "img/img3.png"},  # Example SMILES for ethylamine
    "Mol 4": {"smiles": "C=C1CC(CC(=O)N[C@H]2CCN(C(=O)c3ncccc3SC)C23CC3)C1", "image": "img/img4.png"},
    # Example SMILES for diethyl ether
    "Mol 5": {"smiles": "C=CCS[C@@H](C)CC(=O)OCC", "image": "img/img5.png"}  # Example SMILES for chloroethane
}

datasets = ["Load Custom Dataset"]

models_enabled = ["SELFIES-TED", "MHG-GED", "MolFormer", "SMI-TED"]

fusion_available = ["Concat"]

global log_df
log_df = pd.DataFrame(columns=["Selected Models", "Dataset", "Task", "Result"])


def log_selection(models, dataset, task_type, result, log_df):
    # Append the new entry to the DataFrame
    new_entry = {"Selected Models": str(models), "Dataset": dataset, "Task": task_type, "Result": result}
    updated_log_df = log_df.append(new_entry, ignore_index=True)
    return updated_log_df


# Function to handle evaluation and logging
def save_rep(models, dataset, task_type, eval_output):
    return
def evaluate_and_log(models, dataset, task_type, eval_output):
    task_dic = {'Classification': 'CLS', 'Regression': 'RGR'}
    result = f"{eval_output}"#display_eval(models, dataset, task_type, fusion_type=None)
    result = result.replace(" Score", "")

    new_entry = {"Selected Models": str(models), "Dataset": dataset, "Task": task_dic[task_type], "Result": result}
    new_entry_df = pd.DataFrame([new_entry])

    log_df = pd.read_csv('log.csv', index_col=0)
    log_df = pd.concat([new_entry_df, log_df])

    log_df.to_csv('log.csv')

    return log_df


try:
    log_df = pd.read_csv('log.csv', index_col=0)
except:
    log_df = pd.DataFrame({"":[],
    'Selected Models': [],
    'Dataset': [],
    'Task': [],
    'Result': []
        })
    csv_file_path = 'log.csv'
    log_df.to_csv(csv_file_path, index=False)


# Load images for selection
def load_image(path):
    try:
        return Image.open(smiles_image_mapping[path]["image"])# Image.1open(path)
    except:
        pass



# Function to handle image selection
def handle_image_selection(image_key):
    smiles = smiles_image_mapping[image_key]["smiles"]
    mol_image = smiles_to_image(smiles)
    return smiles, mol_image


def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        qed = QED.qed(mol)
        logp = MolLogP(mol)
        sa = sascorer.calculateScore(mol)
        wt = Descriptors.MolWt(mol)
        return qed, sa, logp, wt
    return None, None, None, None


# Function to calculate Tanimoto similarity
def calculate_tanimoto(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 and mol2:
        # fp1 = FingerprintMols.FingerprintMol(mol1)
        # fp2 = FingerprintMols.FingerprintMol(mol2)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
        return round(DataStructs.FingerprintSimilarity(fp1, fp2), 2)
    return None


#with open("models/selfies_model/bart-2908.pickle", "rb") as input_file:
#    gen_model, gen_tokenizer = pickle.load(input_file)

gen_tokenizer = AutoTokenizer.from_pretrained("ibm/materials.selfies-ted")
gen_model = BartForConditionalGeneration.from_pretrained("ibm/materials.selfies-ted")


def generate(latent_vector, mask):
    encoder_outputs = BaseModelOutput(latent_vector)
    decoder_output = gen_model.generate(encoder_outputs=encoder_outputs, attention_mask=mask,
                                        max_new_tokens=64, do_sample=True, top_k=5, top_p=0.95, num_return_sequences=1)
    selfies = gen_tokenizer.batch_decode(decoder_output, skip_special_tokens=True)
    outs = []
    for i in selfies:
        outs.append(sf.decoder(i.replace("] [", "][")))
    return outs


def perturb_latent(latent_vecs, noise_scale=0.5):
    modified_vec = torch.tensor(np.random.uniform(0, 1, latent_vecs.shape) * noise_scale,
                                dtype=torch.float32) + latent_vecs
    return modified_vec


def encode(selfies):
    encoding = gen_tokenizer(selfies, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    outputs = gen_model.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
    model_output = outputs.last_hidden_state

    """input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    model_output = sum_embeddings / sum_mask"""
    return model_output, attention_mask


# Function to generate canonical SMILES and molecule image
def generate_canonical(smiles):
    s = sf.encoder(smiles)
    selfie = s.replace("][", "] [")
    latent_vec, mask = encode([selfie])
    gen_mol = None
    for i in range(5, 51):
        noise = i / 10
        perturbed_latent = perturb_latent(latent_vec, noise_scale=noise)
        gen = generate(perturbed_latent, mask)
        gen_mol = Chem.MolToSmiles(Chem.MolFromSmiles(gen[0]))
        if gen_mol != Chem.MolToSmiles(Chem.MolFromSmiles(smiles)): break

    if gen_mol:
        # Calculate properties for ref and gen molecules
        ref_properties = calculate_properties(smiles)
        gen_properties = calculate_properties(gen_mol)
        tanimoto_similarity = calculate_tanimoto(smiles, gen_mol)

        # Prepare the table with ref mol and gen mol
        data = {
            "Property": ["QED", "SA", "LogP", "Mol Wt", "Tanimoto Similarity"],
            "Reference Mol": [ref_properties[0], ref_properties[1], ref_properties[2], ref_properties[3],
                              tanimoto_similarity],
            "Generated Mol": [gen_properties[0], gen_properties[1], gen_properties[2], gen_properties[3], ""]
        }
        df = pd.DataFrame(data)

        # Display molecule image of canonical smiles
        mol_image = smiles_to_image(gen_mol)

        return df, gen_mol, mol_image
    return "Invalid SMILES", None, None


# Function to display evaluation score
def display_eval(selected_models, dataset, task_type, downstream, fusion_type):
    result = None

    try:
        downstream_model = downstream.split("*")[0].lstrip()
        downstream_model = downstream_model.rstrip()
        hyp_param = downstream.split("*")[-1].lstrip()
        hyp_param = hyp_param.rstrip()
        hyp_param = hyp_param.replace("nan", "float('nan')")
        params = eval(hyp_param)
    except:
        downstream_model = downstream.split("*")[0].lstrip()
        downstream_model = downstream_model.rstrip()
        params = None




    try:
        if not selected_models:
            return "Please select at least one enabled model."

        if task_type == "Classification":
            global roc_auc, fpr, tpr, x_batch, y_batch
        elif task_type == "Regression":
            global RMSE, y_batch_test, y_prob

        if len(selected_models) > 1:
            if task_type == "Classification":
                #result, roc_auc, fpr, tpr, x_batch, y_batch = fm4m.multi_modal(model_list=selected_models,
                #                                                               downstream_model="XGBClassifier",
                #                                                               dataset=dataset.lower())
                if downstream_model == "Default Settings":
                    downstream_model = "DefaultClassifier"
                    params = None
                result, roc_auc, fpr, tpr, x_batch, y_batch = fm4m.multi_modal(model_list=selected_models,
                                                                                               downstream_model=downstream_model,
                                                                                               params = params,
                                                                                               dataset=dataset)

            elif task_type == "Regression":
                #result, RMSE, y_batch_test, y_prob = fm4m.multi_modal(model_list=selected_models,
                #                                                      downstream_model="XGBRegressor",
                #                                                      dataset=dataset.lower())

                if downstream_model == "Default Settings":
                    downstream_model = "DefaultRegressor"
                    params = None

                result, RMSE, y_batch_test, y_prob, x_batch, y_batch = fm4m.multi_modal(model_list=selected_models,
                                                                      downstream_model=downstream_model,
                                                                      params=params,
                                                                      dataset=dataset)

        else:
            if task_type == "Classification":
                #result, roc_auc, fpr, tpr, x_batch, y_batch = fm4m.single_modal(model=selected_models[0],
                #                                                                downstream_model="XGBClassifier",
                #                                                                dataset=dataset.lower())
                if downstream_model == "Default Settings":
                    downstream_model = "DefaultClassifier"
                    params = None

                result, roc_auc, fpr, tpr, x_batch, y_batch = fm4m.single_modal(model=selected_models[0],
                                                                                downstream_model=downstream_model,
                                                                                params=params,
                                                                                dataset=dataset)

            elif task_type == "Regression":
                #result, RMSE, y_batch_test, y_prob = fm4m.single_modal(model=selected_models[0],
                #                                                       downstream_model="XGBRegressor",
                #                                                       dataset=dataset.lower())

                if downstream_model == "Default Settings":
                    downstream_model = "DefaultRegressor"
                    params = None

                result, RMSE, y_batch_test, y_prob, x_batch, y_batch = fm4m.single_modal(model=selected_models[0],
                                                                       downstream_model=downstream_model,
                                                                       params=params,
                                                                       dataset=dataset)

        if result == None:
            result = "Data & Model Setting is incorrect"
    except Exception as e:
        return f"An error occurred: {e}"
    return f"{result}"


# Function to handle plot display
def display_plot(plot_type):
    fig, ax = plt.subplots()

    if plot_type == "Latent Space":
        global x_batch, y_batch
        ax.set_title("T-SNE Plot")
        # reducer = umap.UMAP(metric='euclidean', n_neighbors=  10, n_components=2, low_memory=True, min_dist=0.1, verbose=False)
        # features_umap = reducer.fit_transform(x_batch[:500])
        # x = y_batch.values[:500]
        # index_0 = [index for index in range(len(x)) if x[index] == 0]
        # index_1 = [index for index in range(len(x)) if x[index] == 1]
        class_0 = x_batch  # features_umap[index_0]
        class_1 = y_batch  # features_umap[index_1]

        """with open("latent_multi_bace.pkl", "rb") as f:
            class_0, class_1 = pickle.load(f)
        """
        plt.scatter(class_1[:, 0], class_1[:, 1], c='red', label='Class 1')
        plt.scatter(class_0[:, 0], class_0[:, 1], c='blue', label='Class 0')

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Dataset Distribution')

    elif plot_type == "ROC-AUC":
        global roc_auc, fpr, tpr
        ax.set_title("ROC-AUC Curve")
        try:
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
        except:
            pass
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc='lower right')

    elif plot_type == "Parity Plot":
        global RMSE, y_batch_test, y_prob
        ax.set_title("Parity plot")

        # change format
        try:
            print(y_batch_test)
            print(y_prob)
            y_batch_test = np.array(y_batch_test, dtype=float)
            y_prob = np.array(y_prob, dtype=float)
            ax.scatter(y_batch_test, y_prob, color="blue", label=f"Predicted vs Actual (RMSE: {RMSE:.4f})")
            min_val = min(min(y_batch_test), min(y_prob))
            max_val = max(max(y_batch_test), max(y_prob))
            ax.plot([min_val, max_val], [min_val, max_val], 'r-')

        except:

            y_batch_test = []
            y_prob = []
            RMSE = None
            print(y_batch_test)
            print(y_prob)





        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')

        ax.legend(loc='lower right')
    return fig


# Predefined dataset paths (these should be adjusted to your file paths)
predefined_datasets = {
    #"BACE": f"./data/bace/train.csv, ./data/bace/test.csv, smiles, Class",
    #"ESOL": f"./data/esol/train.csv, ./data/esol/test.csv, smiles, prop",
}


# Function to load a predefined dataset from the local path
def load_predefined_dataset(dataset_name):
    val = predefined_datasets.get(dataset_name)
    try: file_path = val.split(",")[0]
    except:file_path=False

    if file_path:
        df = pd.read_csv(file_path)
        return df.head(), gr.update(choices=list(df.columns)), gr.update(choices=list(df.columns)), f"{dataset_name.lower()}"
    return pd.DataFrame(), gr.update(choices=[]), gr.update(choices=[]), f"Dataset not found"


# Function to display the head of the uploaded CSV file
def display_csv_head(file):
    if file is not None:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file.name)
        return df.head(), gr.update(choices=list(df.columns)), gr.update(choices=list(df.columns))
    return pd.DataFrame(), gr.update(choices=[]), gr.update(choices=[])


# Function to handle dataset selection (predefined or custom)
def handle_dataset_selection(selected_dataset):
    if selected_dataset == "Custom Dataset":
        # Show file upload fields for train and test datasets if "Custom Dataset" is selected
        return gr.update(visible=True), gr.update(visible=True),  gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


# Function to select input and output columns and display a message
def select_columns(input_column, output_column, train_data, test_data,dataset_name):
    if input_column and output_column:
        return f"{train_data.name},{test_data.name},{input_column},{output_column},{dataset_name}"
    return "Please select both input and output columns."

def set_dataname(dataset_name, dataset_selector ):
    if dataset_selector == "Custom Dataset":
        return f"{dataset_name}"
    return f"{dataset_selector}"

# Function to create model based on user input
def create_model(model_name, max_depth=None, n_estimators=None, alpha=None, degree=None, kernel=None):
    if model_name == "XGBClassifier":
        model = xgb.XGBClassifier(objective='binary:logistic',eval_metric= 'auc', max_depth=max_depth, n_estimators=n_estimators, alpha=alpha)
    elif model_name == "SVR":
        model = SVR(degree=degree, kernel=kernel)
    elif model_name == "Kernel Ridge":
        model = KernelRidge(alpha=alpha, degree=degree, kernel=kernel)
    elif model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Default - Auto":
        model = "Default Settings"
        return f"{model}"
    else:
        return "Model not supported."

    return f"{model_name} * {model.get_params()}"
def model_selector(model_name):
    # Dynamically return the appropriate hyperparameter components based on the selected model
    if model_name == "XGBClassifier":
        return (
            gr.Slider(1, 10, label="max_depth"),
            gr.Slider(50, 500, label="n_estimators"),
            gr.Slider(0.1, 10.0, step=0.1, label="alpha")
        )
    elif model_name == "SVR":
        return (
            gr.Slider(1, 5, label="degree"),
            gr.Dropdown(["rbf", "poly", "linear"], label="kernel")
        )
    elif model_name == "Kernel Ridge":
        return (
            gr.Slider(0.1, 10.0, step=0.1, label="alpha"),
            gr.Slider(1, 5, label="degree"),
            gr.Dropdown(["rbf", "poly", "linear"], label="kernel")
        )
    elif model_name == "Linear Regression":
        return ()  # No hyperparameters for Linear Regression
    else:
        return ()



# Define the Gradio layout
# with gr.Blocks(theme=my_theme) as demo:
with gr.Blocks() as demo:
    with gr.Row():
        # Left Column
        with gr.Column():
            gr.HTML('''
           <div style="background-color: #6A8EAE; color: #FFFFFF; padding: 10px;">
                <h3 style="color: #FFFFFF; margin: 0;font-size: 20px;"> Data & Model Setting</h3>
            </div>
            ''')
            # gr.Markdown("## Data & Model Setting")
            #dataset_dropdown = gr.Dropdown(choices=datasets, label="Select Dat")

            # Dropdown menu for predefined datasets including "Custom Dataset" option
            dataset_selector = gr.Dropdown(label="Select Dataset",
                                           choices=list(predefined_datasets.keys()) + ["Custom Dataset"])
            # Display the message for selected columns
            selected_columns_message = gr.Textbox(label="Selected Columns Info", visible=False)

            with gr.Accordion("Dataset Settings", open=True):
                # File upload options for custom dataset (train and test)
                dataset_name = gr.Textbox(label="Dataset Name", visible=False)
                train_file = gr.File(label="Upload Custom Train Dataset", file_types=[".csv"], visible=False)
                train_display = gr.Dataframe(label="Train Dataset Preview (First 5 Rows)", visible=False, interactive=False)

                test_file = gr.File(label="Upload Custom Test Dataset", file_types=[".csv"], visible=False)
                test_display = gr.Dataframe(label="Test Dataset Preview (First 5 Rows)", visible=False, interactive=False)

                # Predefined dataset displays
                predefined_display = gr.Dataframe(label="Predefined Dataset Preview (First 5 Rows)", visible=False,
                                                  interactive=False)



                # Dropdowns for selecting input and output columns for the custom dataset
                input_column_selector = gr.Dropdown(label="Select Input Column", choices=[], visible=False)
                output_column_selector = gr.Dropdown(label="Select Output Column", choices=[], visible=False)

                #selected_columns_message = gr.Textbox(label="Selected Columns Info", visible=True)

                # When a dataset is selected, show either file upload fields (for custom) or load predefined datasets
                dataset_selector.change(handle_dataset_selection,
                                        inputs=dataset_selector,
                                        outputs=[dataset_name, train_file, train_display, test_file, test_display, predefined_display,
                                                 input_column_selector, output_column_selector])

                # When a predefined dataset is selected, load its head and update column selectors
                dataset_selector.change(load_predefined_dataset,
                                        inputs=dataset_selector,
                                        outputs=[predefined_display, input_column_selector, output_column_selector, selected_columns_message])

                # When a custom train file is uploaded, display its head and update column selectors
                train_file.change(display_csv_head, inputs=train_file,
                                  outputs=[train_display, input_column_selector, output_column_selector])

                # When a custom test file is uploaded, display its head
                test_file.change(display_csv_head, inputs=test_file,
                                 outputs=[test_display, input_column_selector, output_column_selector])

                dataset_selector.change(set_dataname,
                                    inputs=[dataset_name, dataset_selector],
                                    outputs=dataset_name)

                # Update the selected columns information when dropdown values are changed
                input_column_selector.change(select_columns,
                                             inputs=[input_column_selector, output_column_selector, train_file, test_file, dataset_name],
                                             outputs=selected_columns_message)

                output_column_selector.change(select_columns,
                                              inputs=[input_column_selector, output_column_selector, train_file, test_file, dataset_name],
                                              outputs=selected_columns_message)

            model_checkbox = gr.CheckboxGroup(choices=models_enabled, label="Select Model")

            # Add disabled checkboxes for GNN and FNN
            # gnn_checkbox = gr.Checkbox(label="GNN (Disabled)", value=False, interactive=False)
            # fnn_checkbox = gr.Checkbox(label="FNN (Disabled)", value=False, interactive=False)

            task_radiobutton = gr.Radio(choices=["Classification", "Regression"], label="Task Type")

            ####### adding hyper parameter tuning ###########
            model_name = gr.Dropdown(["Default - Auto", "XGBClassifier", "SVR", "Kernel Ridge", "Linear Regression"], label="Select Downstream Model")
            with gr.Accordion("Downstream Hyperparameter Settings", open=True):
                # Create placeholders for hyperparameter components
                max_depth = gr.Slider(1, 20, step=1,visible=False, label="max_depth")
                n_estimators = gr.Slider(100, 5000, step=100, visible=False, label="n_estimators")
                alpha = gr.Slider(0.1, 10.0, step=0.1, visible=False, label="alpha")
                degree = gr.Slider(1, 20, step=1,visible=False, label="degree")
                kernel = gr.Dropdown(choices=["rbf", "poly", "linear"], visible=False, label="kernel")

                # Output textbox
                output = gr.Textbox(label="Loaded Parameters")


            # Dynamically show relevant hyperparameters based on selected model
            def update_hyperparameters(model_name):
                if model_name == "XGBClassifier":
                    return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
                        visible=False), gr.update(visible=False)
                elif model_name == "SVR":
                    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
                        visible=True), gr.update(visible=True)
                elif model_name == "Kernel Ridge":
                    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(
                        visible=True), gr.update(visible=True)
                elif model_name == "Linear Regression":
                    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
                        visible=False), gr.update(visible=False)
                elif model_name == "Default - Auto":
                    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
                        visible=False), gr.update(visible=False)


            # When model is selected, update which hyperparameters are visible
            model_name.change(update_hyperparameters, inputs=[model_name],
                              outputs=[max_depth, n_estimators, alpha, degree, kernel])

            # Submit button to create the model with selected hyperparameters
            submit_button = gr.Button("Create Downstream Model")


            # Function to handle model creation based on input parameters
            def on_submit(model_name, max_depth, n_estimators, alpha, degree, kernel):
                if model_name == "XGBClassifier":
                    return create_model(model_name, max_depth=max_depth, n_estimators=n_estimators, alpha=alpha)
                elif model_name == "SVR":
                    return create_model(model_name, degree=degree, kernel=kernel)
                elif model_name == "Kernel Ridge":
                    return create_model(model_name, alpha=alpha, degree=degree, kernel=kernel)
                elif model_name == "Linear Regression":
                    return create_model(model_name)
                elif model_name == "Default - Auto":
                    return create_model(model_name)

            # When the submit button is clicked, run the on_submit function
            submit_button.click(on_submit, inputs=[model_name, max_depth, n_estimators, alpha, degree, kernel],
                                outputs=output)
            ###### End of hyper param tuning #########

            fusion_radiobutton = gr.Radio(choices=fusion_available, label="Fusion Type")



            eval_button = gr.Button("Train downstream model")
            #eval_button.style(css_class="custom-button-left")

        # Middle Column
        with gr.Column():
            gr.HTML('''
           <div style="background-color: #8F9779; color: #FFFFFF; padding: 10px;">
                <h3 style="color: #FFFFFF; margin: 0;font-size: 20px;"> Downstream Task 1: Property Prediction</h3>
            </div>
            ''')
            # gr.Markdown("## Downstream task Result")
            eval_output = gr.Textbox(label="Train downstream model")

            plot_radio = gr.Radio(choices=["ROC-AUC", "Parity Plot", "Latent Space"], label="Select Plot Type")
            plot_output = gr.Plot(label="Visualization")#, height=250, width=250)

            #download_rep = gr.Button("Download representation")

            create_log = gr.Button("Store log")

            log_table = gr.Dataframe(value=log_df, label="Log of Selections and Results", interactive=False)

            eval_button.click(display_eval,
                              inputs=[model_checkbox, selected_columns_message, task_radiobutton, output, fusion_radiobutton],
                              outputs=eval_output)

            plot_radio.change(display_plot, inputs=plot_radio, outputs=plot_output)


            # Function to gather selected models
            def gather_selected_models(*models):
                selected = [model for model in models if model]
                return selected


            create_log.click(evaluate_and_log, inputs=[model_checkbox, dataset_name, task_radiobutton, eval_output],
                             outputs=log_table)
            #download_rep.click(save_rep, inputs=[model_checkbox, dataset_name, task_radiobutton, eval_output],
            #                 outputs=None)

        # Right Column
        with gr.Column():
            gr.HTML('''
           <div style="background-color: #D2B48C; color: #FFFFFF; padding: 10px;">
                <h3 style="color: #FFFFFF; margin: 0;font-size: 20px;"> Downstream Task 2: Molecule Generation</h3>
            </div>
            ''')
            # gr.Markdown("## Molecular Generation")
            smiles_input = gr.Textbox(label="Input SMILES String")
            image_display = gr.Image(label="Molecule Image", height=250, width=250)
            # Show images for selection
            with gr.Accordion("Select from sample molecules", open=False):
                image_selector = gr.Radio(
                    choices=list(smiles_image_mapping.keys()),
                    label="Select from sample molecules",
                    value=None,
                    #item_images=[load_image(smiles_image_mapping[key]["image"]) for key in smiles_image_mapping.keys()]
                )
                image_selector.change(load_image, image_selector, image_display)
            generate_button = gr.Button("Generate")
            gen_image_display = gr.Image(label="Generated Molecule Image", height=250, width=250)
            generated_output = gr.Textbox(label="Generated Output")
            property_table = gr.Dataframe(label="Molecular Properties Comparison")



            # Handle image selection
            image_selector.change(handle_image_selection, inputs=image_selector, outputs=[smiles_input, image_display])
            smiles_input.change(smiles_to_image, inputs=smiles_input, outputs=image_display)

            # Generate button to display canonical SMILES and molecule image
            generate_button.click(generate_canonical, inputs=smiles_input,
                                  outputs=[property_table, generated_output, gen_image_display])


if __name__ == "__main__":
    demo.launch(share=True)
