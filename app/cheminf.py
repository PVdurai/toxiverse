# this module is outlined
# here: https://flask.palletsprojects.com/en/2.0.x/tutorial/views/
import flask
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for, current_app, send_file
)
from flask_login import current_user, login_required
from app.db_models import User, Dataset, Chemical, QSARModel, KDNNPrediction, Job
import numpy as np
import uuid
from datetime import datetime
import pandas as pd
import ntpath, os
import app.make_new_predictions as knn

from werkzeug.utils import secure_filename
from rdkit import Chem
from rdkit.Chem import PandasTools
import pickle
from random import randint

from sqlalchemy import exc

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as skl_PCA

import plotly
import plotly.express as px
import plotly.graph_objs as go

import json

import app.chem_io as chem_io
import app.machine_learning as ml
import app.config as config

# from app.db import get_db
from app.db_models import User, db

import itertools

bp = Blueprint('cheminf', __name__, url_prefix='/cheminf')



import logging
# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@bp.route('/curator', methods=['GET', 'POST'])
@login_required
def curator():
    """
    Chemical curator page
    """
    from app.curator.curator import Curator
    logger.info(f"User {current_user.id} accessed the curator page.")
    current_user.has_qsar = any(d.qsar_models for d in current_user.datasets)

    if request.method == 'GET':
        task_in_progress = current_user.has_running_tasks(['curate_chems'])

        # If a curator job was pending and it's NOW done, flash and redirect once (so the message sticks).
        if not task_in_progress and session.pop('curator_pending', False):
            flash(
                "Curation complete. The curated dataset is now available in the *Select Dataset* dropdown below. You can also access it in Cheminformatics → Chemical Space Visualization and Cheminformatics → QSAR Builder.",
                "success"
            )
            return redirect(url_for('cheminf.curator'))

        return render_template(
            'cheminf/curator.html',
            user_datasets=list(current_user.datasets),
            user=current_user,
            task_in_progress=task_in_progress
        )

    # POST: enqueue task, mark pending, then redirect to GET
    dataset_selection = request.form['dataset-selection'].strip()
    dup_selection = request.form['duplicate-selection'].strip()
    create_or_replace = request.form['create-or-replace'].strip()
    replace = (create_or_replace == 'replace')

    try:
        current_user.launch_task(
            'curate_chems',
            f'Curating {dataset_selection} chemicals',
            current_user.id,
            dataset_selection,
            dup_selection,
            replace
        )
        db.session.commit()
        session['curator_pending'] = True  # expect completion; used to trigger the flash later
        logger.info(
            f"Task 'Curate' successfully submitted for dataset {dataset_selection} by user {current_user.id}."
        )
    except exc.OperationalError:
        logger.exception("Failed to submit curation job")
        flash("Failed to submit curation job, please try again", "error")

    return redirect(url_for('cheminf.curator'))


@bp.route('/PCA', methods=('GET', 'POST'))
@login_required
def PCA():
    """
    Render a polished 3D PCA scatter for a selected dataset.
    Auto-detects whether 'activity' is binary, categorical, or continuous and
    uses appropriate coloring (discrete vs continuous). Shows PC variance.
    """
    if request.method == 'GET':
        return render_template('cheminf/PCA.html', user_datasets=list(current_user.datasets))

    dataset_selection = (request.form.get('dataset-selection') or "").strip()
    if not dataset_selection:
        flash("Please select a dataset.", "warning")
        return redirect(url_for('cheminf.PCA'))

    # --- Load data for this user's selected dataset ---
    query_statement = (
        db.session.query(Chemical)
        .join(Dataset, Dataset.id == Chemical.dataset_id)
        .filter(Dataset.dataset_name == dataset_selection)
        .filter(Dataset.user_id == current_user.id)
        .statement
    )
    df = pd.read_sql(query_statement, db.session.connection())
    if df.empty:
        flash(f"No records found for dataset '{dataset_selection}'.", "warning")
        return render_template('cheminf/PCA.html', user_datasets=list(current_user.datasets))

    # --- Descriptors ---
    desc_set = ['MolWt', 'TPSA', 'NumRotatableBonds', 'NumHDonors', 'NumHAcceptors', 'MolLogP']
    try:
        descriptors = chem_io.calc_descriptors_from_frame(df, scale=True, desc_set=desc_set)
    except Exception as e:
        flash(f"Failed to compute descriptors: {e}", "danger")
        return render_template('cheminf/PCA.html', user_datasets=list(current_user.datasets))

    if descriptors is None or descriptors.empty:
        flash("No descriptors could be computed for this dataset.", "warning")
        return render_template('cheminf/PCA.html', user_datasets=list(current_user.datasets))

    # --- PCA ---
    try:
        pca_fit = skl_PCA(n_components=3).fit(descriptors)
        pca_scores = pca_fit.transform(descriptors)
    except Exception as e:
        flash(f"PCA failed: {e}", "danger")
        return render_template('cheminf/PCA.html', user_datasets=list(current_user.datasets))

    # Explained-variance percentages
    var = pca_fit.explained_variance_ratio_
    pc1_pct, pc2_pct, pc3_pct = (var[0]*100, var[1]*100, var[2]*100)
    cum_pct = float(var[:3].sum() * 100)

    pca = pd.DataFrame(pca_scores, columns=['PCA1', 'PCA2', 'PCA3'], index=descriptors.index)

    # --- IDs & Activity ---
    if 'compound_id' not in df.columns:
        flash("The dataset is missing required column 'compound_id'.", "danger")
        return render_template('cheminf/PCA.html', user_datasets=list(current_user.datasets))

    df_idx = df.set_index('compound_id')
    # Ensure alignment to avoid KeyErrors if any index mismatch slips in
    common_idx = pca.index.intersection(df_idx.index)
    pca = pca.loc[common_idx]
    df_idx = df_idx.loc[common_idx]

    pca['CMP_ID'] = df_idx.index

    # Decide on activity coloring mode
    color_col = None
    use_continuous = False

    if 'activity' in df_idx.columns:
        activity_series = df_idx['activity'].copy()
        vals = pd.to_numeric(activity_series, errors='coerce')
        unique_vals = set(pd.unique(vals.dropna()))

        if unique_vals and unique_vals.issubset({0, 1}):
            # Binary -> nice labels
            pca['Activity'] = vals.map({1: 'Active', 0: 'Inactive'}).fillna('Unknown').astype('category')
            color_col = 'Activity'
            use_continuous = False
        elif pd.api.types.is_numeric_dtype(vals) and len(unique_vals) > 12:
            # Continuous numeric
            pca['ActivityValue'] = vals.astype(float)
            color_col = 'ActivityValue'
            use_continuous = True
        else:
            # Small discrete set (categorical)
            pca['Activity'] = activity_series.astype(str).fillna('Unknown').astype('category')
            color_col = 'Activity'
            use_continuous = False
    else:
        pca['Activity'] = pd.Categorical(['Unlabeled'] * len(pca))
        color_col = 'Activity'
        use_continuous = False

    # --- Zoomed-in axis ranges ---
    def compute_range(series: pd.Series, q_low=1, q_high=99, pad_frac=0.05):
        a = series.astype(float)
        lo, hi = np.percentile(a, [q_low, q_high])
        span = (hi - lo) if hi > lo else 1.0
        pad = pad_frac * span
        return float(lo - pad), float(hi + pad)

    ranges = {
        'PCA1': compute_range(pca['PCA1']),
        'PCA2': compute_range(pca['PCA2']),
        'PCA3': compute_range(pca['PCA3']),
    }

    # --- Colors ---
    okabe_ito = [
        "#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
    ]

    if use_continuous:
        # continuous colorbar
        fig = px.scatter_3d(
            pca,
            x='PCA1', y='PCA2', z='PCA3',
            color=color_col,  # 'ActivityValue'
            hover_name='CMP_ID',
            hover_data={'PCA1': ':.3f', 'PCA2': ':.3f', 'PCA3': ':.3f', color_col: ':.3g'},
            labels={"PCA1": f"PC1 ({pc1_pct:.1f}%)", "PCA2": f"PC2 ({pc2_pct:.1f}%)", "PCA3": f"PC3 ({pc3_pct:.1f}%)"},
            height=700,
            color_continuous_scale='Viridis'
        )
        fig.update_traces(
            marker=dict(size=6, opacity=0.80, line=dict(width=0.5, color='white')),
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "Value: %{marker.color:.3g}<br>"
                "PC1: %{x:.3f}<br>"
                "PC2: %{y:.3f}<br>"
                "PC3: %{z:.3f}<extra></extra>"
            )
        )
        fig.update_layout(
            coloraxis_colorbar=dict(
                title=dict(text="Activity value", side="top"),
                len=0.90,
                thickness=18,
                x=1.02, xanchor="left",
                y=0.50, yanchor="middle",
                ticks="outside"
            )
        )
    else:
        # categorical legend (includes binary)
        cats = list(pca[color_col].cat.categories) if pd.api.types.is_categorical_dtype(pca[color_col]) else None

        # default mapping
        color_map = {cat: okabe_ito[i % len(okabe_ito)] for i, cat in enumerate(cats or [])}

        # nicer mapping/order for binary
        binary_set = {'Active', 'Inactive', 'Unknown', 'Unlabeled'}
        if cats and set(cats).issubset(binary_set):
            cats = [c for c in ['Active', 'Inactive', 'Unknown', 'Unlabeled'] if c in cats]
            color_map = {
                'Active':   '#F68B33',  # orange
                'Inactive': '#2764B5',  # blue
                'Unknown':  '#d4d4d4',
                'Unlabeled': '#000000'
            }

        # Build figure
        fig = px.scatter_3d(
            pca,
            x='PCA1', y='PCA2', z='PCA3',
            color=color_col,  # 'Activity'
            hover_name='CMP_ID',
            # For categorical, show the category text in hover via customdata
            hover_data={'PCA1': ':.3f', 'PCA2': ':.3f', 'PCA3': ':.3f', color_col: True},
            category_orders={color_col: cats} if cats else None,
            labels={"PCA1": f"PC1 ({pc1_pct:.1f}%)", "PCA2": f"PC2 ({pc2_pct:.1f}%)", "PCA3": f"PC3 ({pc3_pct:.1f}%)"},
            height=700,
            color_discrete_map=color_map if cats else None
        )
        fig.update_traces(
            marker=dict(size=6, opacity=0.80, line=dict(width=0.5, color='white')),
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                f"{color_col}: "+"%{customdata[0]}"+"<br>"
                "PC1: %{x:.3f}<br>"
                "PC2: %{y:.3f}<br>"
                "PC3: %{z:.3f}<extra></extra>"
            )
        )
        # >>> Legend: title on top, items stacked below, top-right
        fig.update_layout(
            legend=dict(
                orientation='v',
                x=0.98, xanchor='right',
                y=0.98, yanchor='top',
                bgcolor='rgba(255,255,255,0.75)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1,
                itemsizing='constant',
                itemwidth=30,
                font=dict(size=11)
            ),
            legend_title=dict(
                text="Activity",
                font=dict(size=12, color='#111')
            ),
        )

    # --- Layout / scene styling (transparent plot; card supplies white) ---
    fig.update_layout(
        template=None,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text=dataset_selection,
            x=0.5, xanchor='center', y=0.95, yanchor='top',
            font=dict(size=16)
        ),
        margin=dict(l=0, r=120, t=40, b=40),   # keep colorbar/legend inside
        scene=dict(
            aspectmode='cube',
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(range=list(ranges['PCA1']), showbackground=False, showline=True,
                       linewidth=2, linecolor='black', gridcolor='#e5e7eb',
                       zeroline=False, tickfont=dict(size=11)),
            yaxis=dict(range=list(ranges['PCA2']), showbackground=False, showline=True,
                       linewidth=2, linecolor='black', gridcolor='#e5e7eb',
                       zeroline=False, tickfont=dict(size=11)),
            zaxis=dict(range=list(ranges['PCA3']), showbackground=False, showline=True,
                       linewidth=2, linecolor='black', gridcolor='#e5e7eb',
                       zeroline=False, tickfont=dict(size=11)),
        ),
        hoverlabel=dict(font_size=12),
        scene_camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
    )

    # --- Serialize & render ---
    pca_plot = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    flash(f"PCA plot is generated for dataset {dataset_selection} and displayed below.", 'success')
    return render_template('cheminf/PCA.html', user_datasets=list(current_user.datasets), pca_plot=pca_plot)




@bp.route('/QSAR-build', methods=('GET', 'POST'))
@login_required
def QSAR_build():
    """Launch QSAR model building tasks."""
    current_user.has_qsar = any(d.qsar_models for d in current_user.datasets)

    if request.method == 'GET':
        task_in_progress = current_user.has_running_tasks(['build_qsar'])

        if not task_in_progress and session.pop('qsar_pending', False):
            # Update job status to finished
            qsar_job_id = session.pop('qsar_job_id', None)
            if qsar_job_id:
                job = Job.query.filter_by(job_id=qsar_job_id).first()
                if job:
                    job.status = 'finished'
                    job.updated_at = datetime.utcnow()
                    db.session.commit()
                    job_link = url_for('toxpro.job_status', job_id=qsar_job_id, _external=True)
                    flash(
                        "Model training complete."
                        f"View your results at: {job_link}",
                        "success"
                    )
                else:
                    flash(
                        "Model training complete."
                        "Use the download buttons beside 'Trained Models' to export CSVs and figures.",
                        "success"
                    )
            else:
                flash(
                    "Model training complete."
                    "Use the download buttons beside 'Trained Models' to export CSVs and figures.",
                    "success"
                )
            return redirect(url_for('cheminf.QSAR_build'))

        has_class = _any_models_of_type("Classification")
        has_reg  = _any_models_of_type("Regression")

        return render_template(
            'cheminf/QSAR-build.html',
            user_datasets=list(current_user.datasets),
            user=current_user,
            task_in_progress=task_in_progress,
            has_class_models=has_class,
            has_reg_models=has_reg
        )

    # ---------- POST: with compatibility checks ----------
    dataset_selection = request.form.getlist('dataset-selection')           # list[str]
    desc_selection    = request.form.getlist('descriptor-selection')        # list[str]
    alg_selection     = request.form.getlist('algorithm-selection')         # list[str]
    type_selection    = request.form.getlist('type-selection')              # radio → single item list

    if not type_selection:
        flash("Please choose Classification or Regression.", "danger")
        return redirect(url_for('cheminf.QSAR_build'))

    chosen_task_type = type_selection[0]  # "Classification" or "Regression"

    # Look up each dataset’s declared type from DB
    ds_types = {
        d.dataset_name: (d.type or "").strip()
        for d in current_user.datasets
    }

    incompatible = []   # [(dataset_name, ds_type, chosen_task_type)]
    compatible_ds = []  # [dataset_name]

    for ds_name in dataset_selection:
        ds_type = ds_types.get(ds_name, "")
        if ds_type == "Binary" and chosen_task_type != "Classification":
            incompatible.append((ds_name, ds_type, chosen_task_type))
            continue
        if ds_type == "Continuous" and chosen_task_type != "Regression":
            incompatible.append((ds_name, ds_type, chosen_task_type))
            continue
        compatible_ds.append(ds_name)

    # Warn and skip incompatible pairs
    if incompatible:
        bad = ", ".join([f"{nm} (declared {tp})" for nm, tp, _ in incompatible])
        flash(
            "Skipped incompatible selection(s): "
            f"{bad}. Build **Classification** for Binary datasets and **Regression** for Continuous datasets.",
            "warning"
        )

    # If nothing left to run
    if not compatible_ds:
        flash("No compatible dataset/type combinations to train. Nothing submitted.", "danger")
        return redirect(url_for('cheminf.QSAR_build'))

    # Create Job record for tracking with web link
    job_id = uuid.uuid4().hex
    job = Job(
        job_id=job_id,
        session_id='qsar_' + uuid.uuid4().hex[:16],  # QSAR doesn't use session folders
        job_type='qsar_build',
        status='queued',
        user_id=current_user.id,
        created_at=datetime.utcnow()
    )
    db.session.add(job)
    db.session.flush()  # Get job.id without committing

    # Generate job link for user
    job_link = url_for('toxpro.job_status', job_id=job_id, _external=True)

    # Launch only compatible jobs
    import itertools
    name_list = []
    for element in itertools.product(*[compatible_ds, desc_selection, alg_selection, [chosen_task_type]]):
        name_list.append('&'.join(element))

    from sqlalchemy import exc
    task_count = 0
    redis_available = hasattr(current_app, 'task_queue') and current_app.task_queue

    for name in name_list:
        dataset, desc, alg, type1 = name.split("&", 3)
        try:
            if redis_available:
                # Try background task
                current_user.launch_task(
                    'build_qsar',
                    f'Building QSAR Model on {name}',
                    current_user.id,
                    dataset,
                    desc,
                    alg,
                    type1,
                    job_id,  # Pass job_id to task
                    len(name_list)  # Pass total number of tasks
                )
                task_count += 1
            else:
                # Redis not available - run synchronously
                from app.tasks import build_qsar
                print(f"Warning: Redis not available. Running QSAR task synchronously for {name}")
                build_qsar(current_user.id, dataset, desc, alg, type1, job_id, len(name_list))
                task_count += 1

        except (ConnectionError, exc.OperationalError) as e:
            print(f"Task submission failed for {name}: {e}")
            flash(f"Failed to submit model for {name}, please try again", "error")
            job.status = 'failed'

    # Update job status
    if job.status != 'failed' and task_count > 0:
        if redis_available:
            job.status = 'running'  # Background tasks will update to 'finished'
            flash(f"QSAR model training started! Bookmark this page to check status anytime within 24 hours.", 'info')
        else:
            job.status = 'finished'  # Already completed synchronously
            flash(f"QSAR model training completed! View your results on this page.", 'success')
        job.updated_at = datetime.utcnow()

    db.session.commit()
    session['qsar_pending'] = True
    session['qsar_job_id'] = job_id  # Store job ID in session

    # Redirect directly to job status page (easier to bookmark)
    return redirect(job_link)


def _clean_label_for_plot(model_name: str) -> str:
    """Strip -Classification / -Regression suffix for figure labels only."""
    if model_name.endswith("-Classification"):
        return model_name[: -len("-Classification")]
    if model_name.endswith("-Regression"):
        return model_name[: -len("-Regression")]
    return model_name


def _collect_metrics_df_for_user(model_type: str, user_id: int):
    """Build DataFrame of metrics for a specific user (for job status page)."""
    import pandas as pd
    import numpy as np
    from app.db_models import User

    if model_type == "Classification":
        metrics_cfg = [
            ("accuracy", "Accuracy"),
            ("f1_score", "F1 Score"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("area_under_roc", "ROC AUC"),
        ]
    else:
        metrics_cfg = [
            ("r2_score", "R2 Score"),
            ("max_error", "Max Error"),
            ("mean_squared_error", "MSE"),
            ("mean_absolute_percentage_error", "MAPE"),
            ("pinball_score", "Pinball Score"),
        ]

    user = User.query.get(user_id)
    if not user:
        return pd.DataFrame(columns=["Model"] + [nice for _, nice in metrics_cfg])

    rows = []
    for d in user.datasets:
        for m in d.qsar_models:
            if (m.type or "").strip() != model_type:
                continue
            if not m.cvresults:
                continue

            row = {"Model": m.name}
            for attr, nice in metrics_cfg:
                val = getattr(m.cvresults, attr, None)
                try:
                    val = float(val) if val is not None else np.nan
                except Exception:
                    val = np.nan
                row[nice] = val
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["Model"] + [nice for _, nice in metrics_cfg])

    df = pd.DataFrame(rows)
    keep_cols = ["Model"] + [c for c in df.columns if c != "Model" and df[c].notna().any()]
    df = df[keep_cols].copy()

    for c in df.columns:
        if c != "Model":
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
    return df


def _collect_metrics_df(model_type: str):
    """
    Build a DataFrame of metrics for either 'Classification' or 'Regression'.
    Columns: ['Model', <metric display names...>] with ONLY that type’s metrics.
    Excludes rows with no cvresults. Values rounded to 2 decimals.
    """
    import pandas as pd
    import numpy as np

    # Per-type metric mapping: (attribute_on_cvresults, "Nice Column Name")
    if model_type == "Classification":
        metrics_cfg = [
            ("accuracy", "Accuracy"),
            ("f1_score", "F1 Score"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("area_under_roc", "ROC AUC"),
        ]
    else:
        metrics_cfg = [
            ("r2_score", "R2 Score"),
            ("max_error", "Max Error"),
            ("mean_squared_error", "MSE"),
            ("mean_absolute_percentage_error", "MAPE"),
            ("pinball_score", "Pinball Score"),
        ]

    rows = []
    from flask_login import current_user
    for d in current_user.datasets:
        for m in d.qsar_models:
            if (m.type or "").strip() != model_type:
                continue
            if not m.cvresults:
                continue

            row = {"Model": m.name}
            for attr, nice in metrics_cfg:
                val = getattr(m.cvresults, attr, None)
                try:
                    val = float(val) if val is not None else np.nan
                except Exception:
                    val = np.nan
                row[nice] = val
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["Model"] + [nice for _, nice in metrics_cfg])

    df = pd.DataFrame(rows)
    keep_cols = ["Model"] + [c for c in df.columns if c != "Model" and df[c].notna().any()]
    df = df[keep_cols].copy()

    # round numeric cols to 2 decimals
    for c in df.columns:
        if c != "Model":
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
    return df


def _clean_label_for_plot(raw):
    """
    Make compact model labels for plotting.

    Examples:
    - 'Supplementary_File_S2-ECFP6-RF-Classification' -> 'ECFP6-RF'
    - '.../bcrp_avg_...-FCFP6-SVM-Regression'        -> 'FCFP6-SVM'
    - 'RDKit-RF'                                      -> 'RDKit-RF'  (unchanged)
    """
    import re
    if raw is None:
        return ""
    s = str(raw)

    # Keep only the "basename" if any path-like bits exist
    s = re.split(r"[\\/]", s)[-1]

    # Split on dashes and remove empty fragments
    parts = [p for p in s.split("-") if p]

    # Drop a trailing task tag if present
    if parts and parts[-1].lower() in {"classification", "regression"}:
        parts = parts[:-1]

    # If there are many tokens, keep only the last two (e.g., 'ECFP6-RF')
    if len(parts) >= 2:
        return "-".join(parts[-2:])
    # Fallback: return whatever remains
    return "-".join(parts)


def _render_heatmap_png(df, is_classification: bool) -> bytes:
    """
    Render a multi-metric line chart (X=Model, Y=Score) to PNG bytes.
    - Classification: clamp Y to ~[0,1] with a little padding
    - Regression: autoscale to data min/max
    - Style: markers, grid, legend titled 'Metrics'
    """
    import io
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty or df.shape[1] <= 1:
        raise ValueError("No metrics available to plot.")

    # X-axis labels (models), optionally cleaned by your helper
    x_labels = df["Model"].map(_clean_label_for_plot) if "Model" in df else None
    if x_labels is None:
        raise ValueError("DataFrame must contain a 'Model' column.")

    # Keep numeric metric columns only
    metrics_df = df.drop(columns=["Model"]).copy()
    for c in metrics_df.columns:
        metrics_df[c] = pd.to_numeric(metrics_df[c], errors="coerce")
    metrics_df = metrics_df.astype(float)

    # Sort by model label for consistent left→right order (optional)
    order = np.argsort(x_labels.str.lower().to_numpy())
    x_labels = x_labels.iloc[order].reset_index(drop=True)
    metrics_df = metrics_df.iloc[order].reset_index(drop=True)

    # --- Plot ---
    plt.figure(figsize=(13, 6))
    x = np.arange(len(x_labels))

    for col in metrics_df.columns:
        y = metrics_df[col].to_numpy(dtype=float)
        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=col.replace("_", " ")
        )

    plt.xticks(x, x_labels, rotation=45, ha="right")
    plt.xlabel("Model")
    plt.ylabel("Score")

    title = "Metrics Across Models"
    plt.title(title, fontsize=20, weight="bold")
    plt.grid(True, linestyle="--", alpha=0.3)

    # Legend inside the plot, transparent background
    plt.legend(
        title="Metrics",
        fontsize=10,
        title_fontsize=11,
        loc="best",          # choose location with most free space
        frameon=False,       # no box
        bbox_to_anchor=(1, 1)  # optional fine-tuning
    )

    # Y-limits
    finite_vals = np.isfinite(metrics_df.to_numpy(dtype=float))
    if finite_vals.any():
        ymin = float(np.nanmin(metrics_df.to_numpy(dtype=float)[finite_vals]))
        ymax = float(np.nanmax(metrics_df.to_numpy(dtype=float)[finite_vals]))
    else:
        ymin, ymax = 0.0, 1.0

    if is_classification:
        # Clamp to [0,1] with a little padding (matches classification scores)
        pad = 0.02
        plt.ylim(max(0.0, 0.0 - pad), min(1.0, 1.0 + pad))
    else:
        # Autoscale with a small margin
        span = max(ymax - ymin, 1e-6)
        pad = 0.05 * span
        plt.ylim(ymin - pad, ymax + pad)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()


def _any_models_of_type(model_type: str) -> bool:
    from flask_login import current_user
    for d in current_user.datasets:
        for m in d.qsar_models:
            if (m.type or "").strip() == model_type and m.cvresults:
                return True
    return False


@bp.route('/QSAR-download-metrics', methods=['GET'])
@login_required
def QSAR_download_metrics():
    """
    Download metrics as CSV or PNG figure.
    Query params:
      type: 'classification' | 'regression'
      format: 'csv' | 'png'
    """
    from flask import request, abort, send_file
    import io

    typ = (request.args.get("type", "") or "").strip().lower()
    fmt = (request.args.get("format", "") or "").strip().lower()

    if typ not in {"classification", "regression"}:
        return abort(400, description="Invalid type.")
    if fmt not in {"csv", "png"}:
        return abort(400, description="Invalid format.")

    model_type = "Classification" if typ == "classification" else "Regression"
    df = _collect_metrics_df(model_type)

    if df.empty or df.shape[1] <= 1:
        return abort(404, description=f"No {model_type} metrics available.")

    if fmt == "csv":
        mem = io.BytesIO()
        df.to_csv(mem, index=False)
        mem.seek(0)
        return send_file(mem, as_attachment=True,
                         download_name=f"{model_type.lower()}_metrics.csv",
                         mimetype="text/csv")

    # fmt == "png"
    png_bytes = _render_heatmap_png(df, is_classification=(model_type == "Classification"))
    return send_file(io.BytesIO(png_bytes), as_attachment=True,
                     download_name=f"{model_type.lower()}_metrics.png",
                     mimetype="image/png")


@bp.route('/download-job-metrics/<int:user_id>')
def download_job_metrics(user_id):
    """Download QSAR metrics for job status page (no login required)."""
    from flask import request, abort, send_file
    import io

    typ = (request.args.get("type", "") or "").strip().lower()
    fmt = (request.args.get("format", "") or "").strip().lower()

    if typ not in {"classification", "regression"}:
        return abort(400, description="Invalid type.")
    if fmt not in {"csv", "png"}:
        return abort(400, description="Invalid format.")

    model_type = "Classification" if typ == "classification" else "Regression"

    # Collect metrics for this user's models
    from app.db_models import User
    user = User.query.get(user_id)
    if not user:
        return abort(404, description="User not found")

    # Get metrics using existing helper function but filtered by user
    df = _collect_metrics_df_for_user(model_type, user_id)

    if df.empty or df.shape[1] <= 1:
        return abort(404, description=f"No {model_type} metrics available.")

    if fmt == "csv":
        mem = io.BytesIO()
        df.to_csv(mem, index=False)
        mem.seek(0)
        return send_file(mem, as_attachment=True,
                         download_name=f"{model_type.lower()}_metrics.csv",
                         mimetype="text/csv")

    # fmt == "png"
    png_bytes = _render_heatmap_png(df, is_classification=(model_type == "Classification"))
    return send_file(io.BytesIO(png_bytes), as_attachment=True,
                     download_name=f"{model_type.lower()}_metrics.png",
                     mimetype="image/png")


@bp.route('/download-model-metrics/<int:model_id>')
def download_model_metrics(model_id):
    """Download a single QSAR model's metrics as CSV."""
    from flask import send_file, abort
    import io
    import csv

    # Fetch model (no login required for shareable job links)
    model = QSARModel.query.get(model_id)
    if not model or not model.cvresults:
        abort(404, description="Model not found")

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header and data
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['Model Name', model.name])
    writer.writerow(['Algorithm', model.algorithm])
    writer.writerow(['Descriptors', model.descriptors])
    writer.writerow(['Type', model.type])
    writer.writerow([''])  # Empty row

    def _fmt_mean_sd(mean, sd):
        if mean is None:
            return ''
        if sd is None:
            return f"{mean:.4f}"
        return f"{mean:.4f} +/- {sd:.4f}"

    if model.type == 'Classification':
        writer.writerow(['Accuracy', _fmt_mean_sd(model.cvresults.accuracy, model.cvresults.accuracy_sd)])
        writer.writerow(['F1 Score', _fmt_mean_sd(model.cvresults.f1_score, model.cvresults.f1_score_sd)])
        writer.writerow(['Precision', _fmt_mean_sd(model.cvresults.precision, model.cvresults.precision_sd)])
        writer.writerow(['Recall', _fmt_mean_sd(model.cvresults.recall, model.cvresults.recall_sd)])
        writer.writerow(['ROC AUC', _fmt_mean_sd(model.cvresults.area_under_roc, model.cvresults.area_under_roc_sd)])
        writer.writerow(['Specificity', _fmt_mean_sd(model.cvresults.specificity, model.cvresults.specificity_sd)])
        writer.writerow(['CCR', _fmt_mean_sd(model.cvresults.correct_classification_rate, model.cvresults.correct_classification_rate_sd)])
        writer.writerow(['Classification threshold', f"{model.cvresults.classification_threshold:.4f}" if model.cvresults.classification_threshold is not None else ''])
    else:  # Regression
        writer.writerow(['R² Score', _fmt_mean_sd(model.cvresults.r2_score, model.cvresults.r2_score_sd)])
        writer.writerow(['Max Error', _fmt_mean_sd(model.cvresults.max_error, model.cvresults.max_error_sd)])
        writer.writerow(['Mean Squared Error', _fmt_mean_sd(model.cvresults.mean_squared_error, model.cvresults.mean_squared_error_sd)])
        writer.writerow(['MAPE', _fmt_mean_sd(model.cvresults.mean_absolute_percentage_error, model.cvresults.mean_absolute_percentage_error_sd)])
        writer.writerow(['Pinball Score', _fmt_mean_sd(model.cvresults.pinball_score, model.cvresults.pinball_score_sd)])

    # Convert to bytes
    output.seek(0)
    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)

    return send_file(mem,
                     as_attachment=True,
                     download_name=f"{model.name}_metrics.csv",
                     mimetype="text/csv")
    
    # Required additional imports near the top of app/app/cheminf.py
# import uuid
# import numpy as np
# from rdkit.Chem.Draw import rdMolDraw2D
# from rdkit.Chem import rdDepictor


def _sanitize_smiles_series(series):
    """Coerce a pandas Series to clean SMILES strings and mark placeholders as missing."""
    import pandas as pd

    s = series.astype(object)

    def _coerce(value):
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        return str(value).strip()

    s = s.map(_coerce)
    return s.replace({'', 'nan', 'NaN', 'NONE', 'None', 'NULL', 'null'}, None)


def _mol_to_svg(mol, width=180, height=130):
    """Return an RDKit SVG string for a molecule for display in Jinja templates."""
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdDepictor

    if mol is None:
        return ""

    mol_for_drawing = Chem.Mol(mol)
    try:
        rdDepictor.Compute2DCoords(mol_for_drawing)
    except Exception:
        pass

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol_for_drawing)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("svg:", "")


def _similarity_fingerprint(mol, radius=3, n_bits=1024):
    """Morgan/ECFP6-like fingerprint used for applicability-domain similarity."""
    from rdkit.Chem import AllChem

    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits,
        useFeatures=False,
    )




def _format_closest_compound_id(compound_id):
    """Display numeric PubChem compound IDs as CID84677 instead of 84677."""
    if compound_id is None:
        return 'Not available'

    value = str(compound_id).strip()
    if not value:
        return 'Not available'

    if value.lower().startswith('cid'):
        return 'CID' + value[3:].strip()

    # Convert values such as 84677 or 84677.0 to CID84677.
    try:
        numeric_value = float(value)
        if numeric_value.is_integer():
            return f'CID{int(numeric_value)}'
    except Exception:
        pass

    if value.isdigit():
        return f'CID{value}'

    return value


def _result_rows_to_model_thresholds(result_rows):
    """Return one threshold row per model for display above the results table."""
    threshold_by_model = {}

    for row in result_rows:
        model = row.get('model')
        threshold = row.get('threshold', 'Not available')
        if model and model not in threshold_by_model:
            threshold_by_model[model] = threshold

    return [
        {'model': model, 'threshold': threshold}
        for model, threshold in threshold_by_model.items()
    ]

def _build_prediction_features(mols_df, descriptor_type):
    """Build model input features using the same descriptor families as QSAR Builder."""
    import numpy as np
    import pandas as pd
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from rdkit.Chem import Descriptors

    descriptor_type = (descriptor_type or '').strip()

    if descriptor_type in ['ECFP6', 'FCFP6']:
        n_bits = 1024
        use_features = descriptor_type == 'FCFP6'
        fps = []
        ids = []

        for mol, compound_id in zip(mols_df['ROMol'], mols_df['compound_id']):
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=3,
                nBits=n_bits,
                useFeatures=use_features,
            )
            arr = np.zeros((n_bits,), dtype=float)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            ids.append(compound_id)

        return pd.DataFrame(fps, index=ids)

    desc_list = [desc[0] for desc in Descriptors.descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)
    descs = []
    ids = []

    for mol, compound_id in zip(mols_df['ROMol'], mols_df['compound_id']):
        if mol is None:
            continue
        descs.append(calc.CalcDescriptors(mol))
        ids.append(compound_id)

    X_predict = pd.DataFrame(descs, index=ids, columns=calc.GetDescriptorNames())
    X_predict = X_predict.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return X_predict


def _get_training_space(qsar_model):
    """Return valid training-set molecules and fingerprints for one QSAR model."""
    from rdkit import Chem
    from app.db_models import Chemical

    training_chemicals = Chemical.query.filter_by(dataset_id=qsar_model.dataset_id).all()
    training_space = []

    for chemical in training_chemicals:
        if not chemical.inchi:
            continue

        mol = Chem.MolFromInchi(chemical.inchi)
        if mol is None:
            continue

        fingerprint = _similarity_fingerprint(mol)
        if fingerprint is None:
            continue

        training_space.append({
            'compound_id': chemical.compound_id or f'training_{chemical.id}',
            'mol': mol,
            'smiles': Chem.MolToSmiles(mol, isomericSmiles=True),
            'fingerprint': fingerprint,
        })

    return training_space


def _leave_one_out_similarity_threshold(training_space, percentile=5):
    """Calculate a model-specific applicability-domain threshold."""
    import numpy as np
    from rdkit import DataStructs

    if len(training_space) < 2:
        return None

    nearest_values = []
    for i, item in enumerate(training_space):
        other_fps = [
            other['fingerprint']
            for j, other in enumerate(training_space)
            if j != i
        ]
        similarities = DataStructs.BulkTanimotoSimilarity(item['fingerprint'], other_fps)
        if similarities:
            nearest_values.append(max(similarities))

    if not nearest_values:
        return None

    return float(np.percentile(nearest_values, percentile))


def _closest_training_match(query_mol, training_space):
    """Find the most similar training-set chemical for a query molecule."""
    import numpy as np
    from rdkit import DataStructs

    query_fp = _similarity_fingerprint(query_mol)
    if query_fp is None or not training_space:
        return None, None

    training_fps = [item['fingerprint'] for item in training_space]
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, training_fps)

    if not similarities:
        return None, None

    best_index = int(np.argmax(similarities))
    return training_space[best_index], float(similarities[best_index])


def _format_prediction(qsar_model, value):
    """Format prediction values for display and CSV export."""
    import numpy as np

    if value is None:
        return 'Not available'

    model_type = (qsar_model.type or '').strip().lower()
    if model_type == 'classification':
        try:
            return 'Active' if int(value) == 1 else 'Inactive'
        except Exception:
            return str(value)

    try:
        if isinstance(value, (float, np.floating)):
            return round(float(value), 4)
    except Exception:
        pass

    return value


def _build_qsar_result_rows(mols_df, model_names, user_id):
    """Run selected models and build rows for the HTML table and CSV download."""
    import pickle
    import pandas as pd
    from app.db_models import QSARModel

    result_rows = []
    prediction_errors = []

    for model_name in model_names:
        qsar_model = QSARModel.query.filter_by(
            user_id=user_id,
            name=model_name,
        ).first()

        if qsar_model is None:
            prediction_errors.append(f"Model '{model_name}' was not found.")
            continue

        try:
            sklearn_model = pickle.loads(qsar_model.sklearn_model)
            X_predict = _build_prediction_features(mols_df, qsar_model.descriptors)
        except Exception as error:
            prediction_errors.append(f"Could not prepare input features for model '{model_name}': {error}")
            continue

        if X_predict.empty:
            prediction_errors.append(f"No valid feature rows were generated for model '{model_name}'.")
            continue

        try:
            if (qsar_model.type or '').strip() == 'Classification' and hasattr(sklearn_model, 'predict_proba'):
                class_threshold = 0.5
                if qsar_model.cvresults and qsar_model.cvresults.classification_threshold is not None:
                    class_threshold = float(qsar_model.cvresults.classification_threshold)

                y_prob = sklearn_model.predict_proba(X_predict)[:, 1]
                y_pred = (y_prob >= class_threshold).astype(int)
            else:
                y_pred = sklearn_model.predict(X_predict)

            prediction_by_id = pd.Series(y_pred, index=X_predict.index)
        except Exception as error:
            prediction_errors.append(f"Prediction failed for model '{model_name}': {error}")
            continue

        training_space = _get_training_space(qsar_model)
        threshold = _leave_one_out_similarity_threshold(training_space)

        for _, query_row in mols_df.iterrows():
            compound_id = query_row['compound_id']
            if compound_id not in prediction_by_id.index:
                continue

            query_mol = query_row['ROMol']
            query_smiles = query_row.get('SMILES') or ''
            closest_match, max_similarity = _closest_training_match(query_mol, training_space)

            if closest_match is None:
                closest_svg = ''
                closest_id = 'Not available'
                closest_smiles = ''
                similarity_display = 'Not available'
                threshold_display = 'Not available'
                domain = 'Not available'
            else:
                closest_svg = _mol_to_svg(closest_match['mol'])
                closest_id = _format_closest_compound_id(closest_match['compound_id'])
                closest_smiles = closest_match['smiles']
                similarity_display = round(max_similarity, 3)

                if threshold is None:
                    threshold_display = 'Not available'
                    domain = 'Not available'
                else:
                    threshold_display = round(threshold, 3)
                    domain = 'Within range' if max_similarity >= threshold else 'Outside range'

            result_rows.append({
                'model': qsar_model.name,
                'query_svg': _mol_to_svg(query_mol),
                'query_smiles': query_smiles,
                'prediction': _format_prediction(qsar_model, prediction_by_id.loc[compound_id]),
                'closest_svg': closest_svg,
                'closest_id': closest_id,
                'closest_smiles': closest_smiles,
                'similarity': similarity_display,
                'threshold': threshold_display,
                'domain': domain,
            })

    return result_rows, prediction_errors


def _result_rows_to_csv_frame(result_rows):
    """Create the mandatory downloadable CSV from the displayed result rows."""
    import pandas as pd

    return pd.DataFrame([
        {
            'Model': row['model'],
            'Query SMILES': row['query_smiles'],
            'Prediction': row['prediction'],
            'Closest match ID': row['closest_id'],
            'Closest match SMILES': row['closest_smiles'],
            'Similarity': row['similarity'],
            'Threshold': row['threshold'],
            'Domain': row['domain'],
        }
        for row in result_rows
    ])


def _save_qsar_csv_for_download(result_rows, base_name, user_id):
    """Save CSV in a user-specific directory and return a simple download filename."""
    import os
    from flask import current_app
    from werkzeug.utils import secure_filename

    download_dir = os.path.join(
        current_app.instance_path,
        'qsar_prediction_downloads',
        f'user_{user_id}'
    )
    os.makedirs(download_dir, exist_ok=True)

    safe_base = secure_filename(base_name or 'qsar_results') or 'qsar_results'
    filename = secure_filename(f'{safe_base}_qsar_results.csv')
    output_path = os.path.join(download_dir, filename)

    csv_df = _result_rows_to_csv_frame(result_rows)
    csv_df.to_csv(output_path, index=False)

    return filename


@bp.route('/QSAR-predict', methods=('GET', 'POST'))
@login_required
def QSAR_predict():
    """Predict activity and display table results with mandatory CSV download."""

    user_qsar_models = list(
        QSARModel.query.filter_by(user_id=current_user.id).order_by(QSARModel.name).all()
    )

    template_context = {
        'user_qsar_models': user_qsar_models,
        'result_rows': [],
        'csv_download_url': None,
        'selected_model_names': [],
        'model_thresholds': [],
    }

    if request.method == 'GET':
        return render_template('cheminf/QSAR-predict.html', **template_context)

    model_names = request.form.getlist('model-selection')
    input_method = request.form.get('input-method', 'file')
    template_context['selected_model_names'] = model_names

    if not model_names:
        flash('Please select at least one QSAR model before submitting.', 'danger')
        return render_template('cheminf/QSAR-predict.html', **template_context)

    error = None
    removed_empty_or_placeholder = 0
    removed_unparsable = 0
    source_rows = 0
    base_name = 'qsar_prediction'

    if input_method == 'text':
        smiles_input = request.form.get('smiles-input', '').strip().splitlines()
        smiles_input = [smiles.strip() for smiles in smiles_input if smiles.strip()]

        if not smiles_input:
            flash('No SMILES provided.', 'danger')
            return render_template('cheminf/QSAR-predict.html', **template_context)

        if len(smiles_input) > 100:
            flash('Maximum number of molecules allowed is 100.', 'danger')
            return render_template('cheminf/QSAR-predict.html', **template_context)

        source_rows = len(smiles_input)
        mols_df = pd.DataFrame({'SMILES': smiles_input})
        mols_df['SMILES'] = _sanitize_smiles_series(mols_df['SMILES'])

        mask_bad = mols_df['SMILES'].isna()
        if mask_bad.any():
            removed_empty_or_placeholder = int(mask_bad.sum())
            mols_df = mols_df.loc[~mask_bad].copy()

        PandasTools.AddMoleculeColumnToFrame(mols_df, smilesCol='SMILES')
        mask_fail = mols_df['ROMol'].isna()
        if mask_fail.any():
            removed_unparsable = int(mask_fail.sum())
            mols_df = mols_df.loc[~mask_fail].copy()

        base_name = 'pasted_smiles'

    else:
        uploaded_file = request.files.get('predict-file')
        smiles_col = request.form.get('smiles-column', '').strip() or 'SMILES'

        if not uploaded_file or uploaded_file.filename == '':
            error = 'No CSV or SDF file was attached.'
        else:
            file_ext = uploaded_file.filename.rsplit('.', 1)[-1].lower()
            if file_ext not in ['csv', 'sdf']:
                error = 'Only CSV or SDF files are accepted.'

        if error:
            flash(error, 'danger')
            return render_template('cheminf/QSAR-predict.html', **template_context)

        uploaded_path = os.path.join(current_app.instance_path, secure_filename(uploaded_file.filename))
        uploaded_file.save(uploaded_path)
        base_name = os.path.splitext(os.path.basename(uploaded_file.filename))[0]

        try:
            if file_ext == 'csv':
                df_tmp = pd.read_csv(uploaded_path, dtype={smiles_col: str})
                source_rows = df_tmp.shape[0]

                if smiles_col not in df_tmp.columns:
                    flash(f"SMILES column '{smiles_col}' not found in the uploaded CSV.", 'danger')
                    return render_template('cheminf/QSAR-predict.html', **template_context)

                df_tmp[smiles_col] = _sanitize_smiles_series(df_tmp[smiles_col])
                mask_bad = df_tmp[smiles_col].isna()
                if mask_bad.any():
                    removed_empty_or_placeholder = int(mask_bad.sum())
                    df_tmp = df_tmp.loc[~mask_bad].copy()

                if df_tmp.empty:
                    flash('All input rows were empty or invalid for SMILES.', 'danger')
                    return render_template('cheminf/QSAR-predict.html', **template_context)

                PandasTools.AddMoleculeColumnToFrame(df_tmp, smilesCol=smiles_col)
                mask_fail = df_tmp['ROMol'].isna()
                if mask_fail.any():
                    removed_unparsable = int(mask_fail.sum())
                    df_tmp = df_tmp.loc[~mask_fail].copy()

                if smiles_col != 'SMILES':
                    df_tmp['SMILES'] = df_tmp[smiles_col]

                mols_df = df_tmp

            else:
                suppl = Chem.SDMolSupplier(
                    uploaded_path,
                    sanitize=False,
                    removeHs=False,
                    strictParsing=False
                )
                rows = []
                source_rows = 0

                smiles_property_names = (
                    'SMILES',
                    'Smiles',
                    'smiles',
                    'CANONICAL_SMILES',
                    'Canonical_SMILES',
                    'canonical_smiles',
                )

                for mol in suppl:
                    source_rows += 1

                    if mol is None:
                        removed_unparsable += 1
                        continue

                    try:
                        props = mol.GetPropsAsDict(includePrivate=False, includeComputed=False)
                        smiles = ''

                        # Case 1: normal SDF with atom/bond block.
                        if mol.GetNumAtoms() > 0:
                            try:
                                Chem.SanitizeMol(mol)

                                try:
                                    mol = Chem.RemoveHs(mol)
                                except Exception:
                                    pass

                                smiles = Chem.MolToSmiles(mol, isomericSmiles=True).strip()
                            except Exception:
                                smiles = ''

                        # Case 2: sample/prediction SDF with empty mol block but SMILES stored as a property.
                        if not smiles:
                            for prop_name in smiles_property_names:
                                if mol.HasProp(prop_name):
                                    smiles = str(mol.GetProp(prop_name)).strip()
                                    break

                        # Extra fallback in case RDKit changes property-name capitalization.
                        if not smiles:
                            for key, value in props.items():
                                if str(key).strip().lower() == 'smiles':
                                    smiles = str(value).strip()
                                    break

                        if not smiles:
                            removed_unparsable += 1
                            continue

                        clean_mol = Chem.MolFromSmiles(smiles, sanitize=True)

                        if clean_mol is None or clean_mol.GetNumAtoms() == 0:
                            removed_unparsable += 1
                            continue

                        clean_smiles = Chem.MolToSmiles(clean_mol, isomericSmiles=True)

                    except Exception:
                        removed_unparsable += 1
                        continue

                    props['ROMol'] = clean_mol
                    props['SMILES'] = clean_smiles
                    rows.append(props)

                if not rows:
                    flash(
                        'No valid molecules were found in the SDF. The file must contain either '
                        'a valid atom/bond structure block or a valid SMILES property.',
                        'danger'
                    )
                    return render_template('cheminf/QSAR-predict.html', **template_context)

                mols_df = pd.DataFrame(rows)
        finally:
            if os.path.exists(uploaded_path):
                os.remove(uploaded_path)

    if mols_df.empty:
        flash('No valid chemicals found after filtering invalid or unparsable SMILES.', 'danger')
        return render_template('cheminf/QSAR-predict.html', **template_context)

    if mols_df.shape[0] > 100:
        flash('Maximum number of molecules allowed is 100 after filtering.', 'danger')
        return render_template('cheminf/QSAR-predict.html', **template_context)

    mols_df = mols_df.reset_index(drop=True)
    mols_df['compound_id'] = [f'mol_{i + 1}' for i in range(mols_df.shape[0])]
    mols_df['inchi'] = [Chem.MolToInchi(mol) for mol in mols_df['ROMol']]

    if removed_empty_or_placeholder or removed_unparsable:
        messages = []
        if removed_empty_or_placeholder:
            messages.append(
                f'{removed_empty_or_placeholder} row(s) with empty/placeholder SMILES were ignored.'
            )
        if removed_unparsable:
            messages.append(
                f'{removed_unparsable} row(s) had SMILES/molecules that RDKit could not parse and were ignored.'
            )
        messages.append(f'Processed {mols_df.shape[0]} molecule(s) out of {source_rows}.')
        flash(' '.join(messages), 'warning')

    result_rows, prediction_errors = _build_qsar_result_rows(
        mols_df=mols_df,
        model_names=model_names,
        user_id=current_user.id,
    )

    for prediction_error in prediction_errors:
        flash(prediction_error, 'danger')

    if not result_rows:
        flash('No predictions were generated. Check the selected models and input structures.', 'danger')
        return render_template('cheminf/QSAR-predict.html', **template_context)

    csv_filename = _save_qsar_csv_for_download(
        result_rows=result_rows,
        base_name=base_name,
        user_id=current_user.id,
    )

    template_context['result_rows'] = result_rows
    template_context['model_thresholds'] = _result_rows_to_model_thresholds(result_rows)
    template_context['csv_download_url'] = url_for(
        'cheminf.QSAR_predict_download_csv',
        filename=csv_filename,
    )

    flash('Prediction results are displayed below. Use the Download CSV button to save the results.', 'success')
    return render_template('cheminf/QSAR-predict.html', **template_context)


@bp.route('/QSAR-predict/download/<path:filename>')
@login_required
def QSAR_predict_download_csv(filename):
    """Download the CSV generated by QSAR Predictor."""
    safe_filename = os.path.basename(filename)

    # Each user's file is saved in a user-specific folder. This keeps the
    # downloaded filename clean, for example pasted_smiles_qsar_results.csv,
    # without exposing user IDs or random UUIDs in the file name.
    download_dir = os.path.join(
        current_app.instance_path,
        'qsar_prediction_downloads',
        f'user_{current_user.id}'
    )

    if not os.path.exists(os.path.join(download_dir, safe_filename)):
        flask.abort(404)

    return flask.send_from_directory(
        download_dir,
        safe_filename,
        as_attachment=True,
        mimetype='text/csv',
        download_name=safe_filename,
    )


@bp.route('/task-table-partial')
@login_required
def task_table_partial():
    return render_template('toxpro/_task_table.html', user=current_user)



# @bp.route('/kDNN', methods=['GET','POST'])
# def kDNN():
#     """
#     displays the kDNN homepage
#
#     """
#     return render_template('cheminf/kDNN.html')

@bp.route('/KDNN-predict', methods=['GET', 'POST'])
@login_required
def KDNN_predict():
    """
    predict results based on prebuilt kDNN model
    """
    current_user.has_kdnn_prediction = any(d for d in current_user.kdnn_prediction)
    if request.method == 'GET':
        return render_template('cheminf/kDNN.html', user_kdnn_predictions=list(current_user.kdnn_prediction),
                               user=current_user)

    error = None
    input_selection = request.form['input-selection'].strip()

    if input_selection == 'manual':
        smiles_input = request.form.getlist('smiles-input')[0][:-1].split(',')
        print(smiles_input[0])
        if len(smiles_input) == 1 and smiles_input[0] == '':
            error = 'No compounds were put in'

        else:
            id = list(range(1, len(smiles_input) + 1))
            compound_id = ["ID-" + str(number) for number in id]
            df = pd.DataFrame(data={'Compound': compound_id,
                                    'SMILES': smiles_input})

            PandasTools.AddMoleculeColumnToFrame(df, smilesCol='SMILES', molCol='ROMol', includeFingerprints=False)
            directory = config.Config.DATA_DIR
            PandasTools.WriteSDF(df, os.path.join(directory, 'df.sdf'), molColName='ROMol',
                                 properties=('Compound', 'SMILES'))
            df1 = PandasTools.LoadSDF(os.path.join(directory, 'df.sdf')).drop(columns=['ID'])
            print(df1)
            name = f'compound set {randint(1, 100)}'

            try:
                predicted_values = knn.make_knn_prediction(dataframe=df1, dataframe_name='df',
                                                           identifier_predict='Compound')
                predicted_values = predicted_values.reset_index().rename(columns={'Identifier': 'Compound'})
                print(predicted_values)
                predicted_values = predicted_values.merge(df1, on='Compound')

            except exc.OperationalError as error:
                flash("Failed to submit request, please try again", 'error')

            # if output_type == 'SDF':
            #     download_file = os.path.join(current_app.instance_path + f"{name}.sdf")
            #     # os.remove(os.path.join(directory, 'df.sdf'))
            #     PandasTools.WriteSDF(predicted_values, download_file,
            #                          properties=predicted_values.drop('ROMol', axis=1).columns)
            #     return flask.send_file(download_file,
            #                            as_attachment=True,
            #                            download_name=f"{name}_predicted.sdf")

            # elif output_type == 'CSV':
            import io
            mem = io.BytesIO()
            mem.write(predicted_values.drop(['index'], axis=1).to_csv().encode())
            mem.seek(0)
            return flask.send_file(
                mem,
                as_attachment=True,
                download_name=f"{name}_predicted.csv",
                mimetype="text/plain",
                )

    else:
        sdfile = request.files['predict-file']
        identifier_column = request.form['identifier-column'].strip()
        class_column = request.form['class-column'].strip()

        if not sdfile:
            error = "No SDFile was attached."

        if sdfile and not sdfile.filename.rsplit('.', 1)[1] in ['csv', 'sdf']:
            error = "The file is not an SDF or CSV file"

        if sdfile:
            file_name = sdfile.filename.rsplit('.', 1)[0]
            sdfile.save(os.path.join(config.Config.DATA_DIR, secure_filename(sdfile.filename)))

            user_uploaded_dataframe = \
            PandasTools.LoadSDF(os.path.join(config.Config.DATA_DIR, secure_filename(sdfile.filename)),
                                smilesName='SMILES')[[identifier_column, class_column, 'SMILES']]
            user_uploaded_dataframe = user_uploaded_dataframe.rename(columns={class_column: 'Actual Class'})

            try:
                current_user.launch_task('kdnn_predict',
                                         f'Predicting uploaded compound set {file_name} by k-DNN model',
                                         current_user.id,
                                         user_uploaded_dataframe,
                                         file_name,
                                         identifier_column)

            except exc.OperationalError as err:
                flash("Failed to submit request, please try again", 'error')

        db.session.commit()
        # os.remove(os.path.join(config.Config.DATA_DIR, secure_filename(sdfile.filename)))

    if error:
        flash(error, 'danger')

    return redirect(url_for('cheminf.KDNN_predict'))


@bp.route('/download-predictions', methods=['POST'])
@login_required
def download_predictions():
    prediction_name = request.form['prediction-selection'].strip()
    do_what = request.form['action']

    if do_what == 'Download Results':
        kdnn_prediction = KDNNPrediction.query.filter_by(name=prediction_name).first()
        prediction_table = pickle.loads(kdnn_prediction.prediction)

        import io
        mem = io.BytesIO()
        mem.write(prediction_table.to_csv().encode())
        mem.seek(0)
        return send_file(
            mem,
            as_attachment=True,
            download_name=f"{prediction_name}.csv",
            mimetype="text/plain",
        )


if __name__ == '__main__':
    from app import create_app

    app = create_app()
    # app.app_context().push()
    query_statement = db.session.query(Dataset).innerjoin(Dataset,
                                                          Dataset.id == Chemical.dataset_id) \
        .filter_by(dataset_name=1, user_id=1) \
        .first()
    print(query_statement)
