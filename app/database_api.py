from flask import Blueprint, request, jsonify, session, flash, render_template
from flask_login import login_required, current_user
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase

from app.db_models import db
import app.config as config
from app.db_models import User, Dataset, Chemical, QSARModel, Bioprofile, KDNNPrediction

from app.master_db import get_database, make_query, get_master

import sys, os, pickle
import pandas as pd


def _clean_endpoint_column(df: pd.DataFrame, col: str) -> pd.Series:
    s = (
        df[col]
        .astype(str)
        .str.strip()
        .replace({
            '': pd.NA,
            'NA': pd.NA,
            'N/A': pd.NA,
            'na': pd.NA,
            'NaN': pd.NA,
            'nan': pd.NA,
            '.': pd.NA,
            '-': pd.NA,
            '--': pd.NA,
        })
    )
    return pd.to_numeric(s, errors='coerce')  # non-numeric -> NaN



TOXICITY_ENDPOINT_INFO = pd.read_csv('data/toxicity-endpoint-info.csv', index_col=0)

bp = Blueprint('api', 'api', url_prefix='/api')


@bp.route('/datasets/<dataset_selection>')
def get_dataset_overview(dataset_selection):
    query_statement = db.session.query(Chemical).join(Dataset,
                                                      Dataset.id == Chemical.dataset_id) \
        .filter(Dataset.dataset_name == dataset_selection) \
        .filter(Dataset.user_id == current_user.id).statement
    df = pd.read_sql(query_statement, db.session.connection())

    actives = df['activity'].sum()
    inactives = df.shape[0] - actives

    data = {
        'name': dataset_selection,
        'actives': int(actives),
        'inactives': int(inactives)
    }
    return jsonify(results=data)


@bp.route('/dataset-data')
def get_dataset_data():
    dataset_selection = request.args.get("datasetSelection")
    search = request.args.get('search[value]')

    query = Dataset.query.filter(Dataset.dataset_name == dataset_selection) \
        .filter(Dataset.user_id == current_user.id) \
        .one().get_chemicals()

    total_chemicals = query.count()

    if search:
        query = query.filter(db.or_(
            Chemical.compound_id.like(f'%{search}%'),
            Chemical.activity.like(f'%{search}%')
        ))

    total_filtered = query.count()

    # pagination
    start = request.args.get('start', type=int)
    length = request.args.get('length', type=int)
    query = query.offset(start).limit(length)

    return {
        "data": [chemical.to_dict(structure_as_svg=True) for chemical in query.all()],
        'recordsFiltered': total_filtered,
        'recordsTotal': total_chemicals,
        'draw': request.args.get('draw', type=int)
    }


@bp.route('/get_assays')
def get_assays():
    profile_selection = request.args.get("profileSelection")

    bioprofile = Bioprofile.query.filter_by(name=profile_selection).first()
    active_assays = pickle.loads(bioprofile.active_assay_table).to_dict(orient='records')

    return {'data': active_assays}


@bp.route('/get_kdnn_predictions')
def get_kdnn_predictions():
    prediction_selection = request.args.get("predictionSelection")

    kdnn_prediction = KDNNPrediction.query.filter_by(name=prediction_selection).first()
    results = pickle.loads(kdnn_prediction.prediction).to_dict(orient='records')

    return {'data': results}


@bp.route('/tox-ep')
def get_toxicity_endpoint():
    """Returns a Plotly histogram trace for the selected endpoint."""
    endpoint_selection = request.args.get("endpointSelection")
    ep_info = TOXICITY_ENDPOINT_INFO.set_index('Endpoint').loc[endpoint_selection].to_dict()

    df = get_database(ep_info['Dataset'])
    df = df[['Master-ID', 'CleanedInChI', endpoint_selection]]
    df[endpoint_selection] = pd.to_numeric(df[endpoint_selection], errors='coerce')
    df = df.dropna(subset=[endpoint_selection])

    # Optional: remove duplicates if needed
    df = df.drop_duplicates(subset='Master-ID')

    trace = {
        'x': df[endpoint_selection].tolist(),
        'type': 'histogram',
        'name': endpoint_selection,
        'marker': {
            'color': 'rgba(100,150,250,0.7)',
            'line': {
                'color': 'rgba(8,48,107,1.0)',
                'width': 1.5
            }
        }
    }

    return jsonify([trace])


@bp.route('/tox-pca-data')
def get_pca_data():
    import numpy as np

    endpoint_selection = request.args.get("endpointSelection")
    ep = TOXICITY_ENDPOINT_INFO.set_index('Endpoint').loc[endpoint_selection].to_dict()
    curated_table = ep['Dataset']

    # --- Load PCA coordinates (all points) ---
    pca_df = make_query("SELECT [Master-ID], PCA1, PCA2, PCA3 FROM chemical_space")

    # --- Membership and endpoint info ---
    in_ids = set(make_query(f"SELECT [Master-ID] FROM [{curated_table}]")['Master-ID'])
    cur_df = get_database(curated_table)[['Master-ID', endpoint_selection]].copy()
    cur_df[endpoint_selection] = _clean_endpoint_column(cur_df, endpoint_selection)
    has_ep = (
        cur_df.groupby('Master-ID')[endpoint_selection]
        .apply(lambda s: s.notna().any())
        .reset_index(name='has_endpoint')
    )

    pca_df = pca_df.merge(has_ep, on='Master-ID', how='left')
    pca_df['has_endpoint'] = pca_df['has_endpoint'].fillna(False)
    pca_df['in_dataset'] = pca_df['Master-ID'].isin(in_ids)

    # --- Assign solid colors and sizes (no rings) ---
    def _color_size(row):
        if not row['in_dataset']:
            return ('rgba(20, 40, 186, 0.6)', 2.5)   # faint blue small
        if row['has_endpoint']:
            return ('rgba(201, 40, 0, 0.9)', 5)      # red medium
        return ('rgba(128,128,128,0.7)', 3.5)        # gray medium

    color_size = pca_df.apply(lambda r: _color_size(r), axis=1)
    colors = [c for c, _ in color_size]
    sizes  = [s for _, s in color_size]

    # --- Single clean trace (no legend, no rings) ---
    trace = {
        'x': pca_df['PCA1'].tolist(),
        'y': pca_df['PCA2'].tolist(),
        'z': pca_df['PCA3'].tolist(),
        'type': 'scatter3d',
        'mode': 'markers',
        'marker': {
            'size': sizes,
            'color': colors,
            'opacity': 0.9,
            'line': {'width': 0}
        },
        'text': pca_df['Master-ID'].astype(str).tolist(),
        'hoverinfo': 'text',   # show only ID
        'showlegend': False
    }

    # --- Axis range helper (zoomed in 1–99%) ---
    def _pr(series, ql=1, qh=99, pad=0.05):
        vals = pd.to_numeric(series, errors='coerce').astype(float).dropna()
        lo, hi = np.percentile(vals, [ql, qh]) if not vals.empty else (0, 1)
        span = hi - lo or 1.0
        return float(lo - pad*span), float(hi + pad*span)

    xr = _pr(pca_df['PCA1'])
    yr = _pr(pca_df['PCA2'])
    zr = _pr(pca_df['PCA3'])

    # --- Clean minimal layout ---
    layout = {
        'template': None,
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'margin': {'l': 0, 'r': 60, 't': 10, 'b': 30},
        'scene': {
            'aspectmode': 'cube',
            'bgcolor': 'rgba(0,0,0,0)',
            'xaxis': {'title': 'PC1', 'range': list(xr), 'showbackground': False,
                      'gridcolor': '#e5e7eb', 'linecolor': 'black', 'linewidth': 2, 'tickfont': {'size': 11}},
            'yaxis': {'title': 'PC2', 'range': list(yr), 'showbackground': False,
                      'gridcolor': '#e5e7eb', 'linecolor': 'black', 'linewidth': 2, 'tickfont': {'size': 11}},
            'zaxis': {'title': 'PC3', 'range': list(zr), 'showbackground': False,
                      'gridcolor': '#e5e7eb', 'linecolor': 'black', 'linewidth': 2, 'tickfont': {'size': 11}},
            'camera': {'eye': {'x': 1.4, 'y': 1.3, 'z': 0.9}}
        },
        'hoverlabel': {'font_size': 12},
        'showlegend': False
    }

    return jsonify({'data': [trace], 'layout': layout})


@bp.route('/update-pca')
def update_pca():
    endpoint_selection = request.args.get("endpointSelection")
    with bp.test_request_context(query_string={'endpointSelection': endpoint_selection}):
        return get_pca_data()


@bp.route('/tox-bioprofile')
def get_bioprofile_data():
    endpoint_selection = request.args.get("endpointSelection", None)

    ep = TOXICITY_ENDPOINT_INFO.set_index('Endpoint').loc[endpoint_selection].to_dict()
    dataset = ep['Dataset']
    df = pd.read_csv(f'data/Bioassays/{dataset}+{endpoint_selection}.csv').to_dict(orient='records')

    # def confusion_matrix(df, activity_class, dataset_name):
    #     """ this function calculates the confusion matrix for an assay, toxicity pair """
    #     df[activity_class] = pd.to_numeric(df[activity_class], errors='coerce')
    #     df = df[df[activity_class].notnull()]
    #
    #     tps = ((df[activity_class] == 1) & (df.Activity_Transformed == 1)).sum()
    #     fps = ((df[activity_class] == 0) & (df.Activity_Transformed == 1)).sum()
    #     tns = ((df[activity_class] == 0) & (df.Activity_Transformed == 0)).sum()
    #     fns = ((df[activity_class] == 1) & (df.Activity_Transformed == 0)).sum()
    #
    #     return tps, fps, tns, fns
    #
    # bioprofile = pd.read_csv(os.path.join(config.Config.BIOPROFILE_DIR, f"{dataset}+{endpoint_selection}.csv"))
    # med = (
    #     bioprofile[['Master-ID', endpoint_selection]]
    #     .drop_duplicates()
    #     [endpoint_selection]
    #     .median()
    # )
    # bioprofile['activity'] = bioprofile[endpoint_selection].copy()
    # if dataset in ['LD50_curated']:
    #     bioprofile.loc[bioprofile[endpoint_selection] < med, 'activity'] = 1
    #     bioprofile.loc[bioprofile[endpoint_selection] >= med, 'activity'] = 0
    # else:
    #     bioprofile.loc[bioprofile[endpoint_selection] < med, 'activity'] = 0
    #     bioprofile.loc[bioprofile[endpoint_selection] >= med, 'activity'] = 1
    #
    # matrix = (
    #     bioprofile
    #     .groupby('AID')
    #     .apply(lambda x: confusion_matrix(x, endpoint_selection, dataset))
    #     .apply(pd.Series)
    #     .set_axis(['TP', 'FP', 'TN', 'FN'], axis=1)
    #     .reset_index()
    #     .sort_values('TP', ascending=False)
    # )
    # matrix['PPV'] = matrix.TP / (matrix.TP + matrix.FP)
    # matrix['Sensitivity'] = matrix.TP / (matrix.TP + matrix.FN)
    # bioprofile = pd.merge(matrix, bioprofile, on='AID', how='inner')
    #
    # PCA_DF = make_query('select [Master-ID], PCA1, PCA2, PCA3'
    #                     ' from chemical_space')
    #
    # CID_DF = make_query('select [Master-ID], [CID]'
    #                     ' from cid_lookup')
    #
    # pca = PCA_DF.merge(CID_DF, on='Master-ID', how='inner').join(
    #     bioprofile[['CID', 'activity']].drop_duplicates().set_index('CID'))
    # pca['CIDs'] = pca.index
    # pca = pd.merge(pca, bioprofile, on='Master-ID', how='inner')
    #
    # bio_info = pd.read_table(config.Config.BIOASSAYS)
    # biodict = dict(zip(bio_info['AID'], bio_info['BioAssay Name']))
    # pca['BioAssay Name'] = pca['AID'].map(biodict)
    #
    # table = pca.groupby(['AID', 'BioAssay Name'])['Activity_Transformed'] \
    #     .value_counts() \
    #     .unstack(fill_value=0) \
    #     .rename(columns={-1.0: 'Inactive', 0.0: 'Inconclusive', 1.0: 'Active'})
    # table['Active rate'] = table['Active'] / (table['Active'] + table['Inactive'])
    # table = pd.DataFrame(table.to_records())
    # top_assays = table.sort_values(by=['Active rate'], ascending=False).head(20).to_dict(orient='records')

    return {'data': df}
