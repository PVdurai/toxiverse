from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, jsonify, send_from_directory, current_app,
    send_file, abort, Response
)
from werkzeug.utils import secure_filename

from rdkit import Chem
from rdkit.Chem import PandasTools

import plotly, itertools
import plotly.express as px
import json, os, ntpath, pickle, uuid
from datetime import datetime

from app.db_models import User, Dataset, Chemical, db, Permission, Bioprofile, Job
import app.master_db as master_db
from flask_login import current_user, login_required, login_manager
from sqlalchemy import exc
import pandas as pd
from plotly.graph_objs import *
from functools import wraps
from app import bioprofile as bpp
import app.pubchem as pc
 
from flask import send_file
import os
from flask import send_file, jsonify
from sklearn.feature_selection import mutual_info_classif

from app.session_manager import SessionManager

bp = Blueprint('toxpro', __name__)

TOXICITY_ENDPOINT_INFO = pd.read_csv('data/toxicity-endpoint-info.csv', index_col=0)

# Path to the project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TMP_FILES_DIR = os.path.join(ROOT_DIR, 'data/tmp')


# Path to the sample_files directory
SAMPLE_FILES_DIR = os.path.join(ROOT_DIR, 'data/sample_files') 

def permission_required(permission):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.can(permission):
                abort(403)
                return f(*args, **kwargs)

        return decorated_function()

    return decorator


def advanced_required(f):
    return permission_required(Permission.ADVANCED)(f)


# this is necessary for declaring
# variables that are available across
# all templates: https://stackoverflow.com/questions/26498689/flask-jinja-pass-data-to-a-base-template-all-templates
# @bp.context_processor
# def inject_nms():
#     #

@bp.route('/', methods=['GET'])
def index():
    """
    displays the homepage

    """
    return render_template('toxpro/home.html')



@bp.route('/tutorial', methods=['GET'])

def tutorial():
    return render_template('toxpro/tutorial.html')

@bp.route('/sourceData', methods=['GET'])

def sourceData():
    return render_template('toxpro/sourceData.html')

@bp.route('/about', methods=['GET'])
def about():
    """
    displays the about page

    """
    return render_template('toxpro/about.html')


@bp.route('/tasks', methods=['GET'])
@login_required
def tasks():
    """
    displays the tasks page

    """
    return render_template('toxpro/tasks.html', user=current_user)


@bp.route('/contact', methods=['GET'])
def contact():
    """
    displays the contact homepage

    """
    return render_template('toxpro/contact.html')


@bp.route('/datasets', methods=['GET'])
@login_required
def datasets():
    """
    Datasets page (Upload / Retrieve).
    Only show/refresh for PubChem import tasks.
    """
    return render_template(
        'toxpro/datasets.html',
        user_datasets=list(current_user.datasets),
        user=current_user,
        task_in_progress=current_user.has_running_tasks(['add_pubchem_data'])  # page-specific
    )


@bp.route('/upload_dataset', methods=['POST'])
@login_required
def upload_dataset():
    """
    Uploads a dataset (CSV or SDF) after validating column names and entries.
    Adds warnings if declared dataset type doesn't match observed values.
    """
    sdfile = request.files['compound_file']

    # Keep raw user input first; do not force defaults yet
    activity_col = request.form['act-id-property'].strip()
    compound_id_col = request.form['cmp-id-property'].strip()
    smiles_col = request.form['smiles-id-property'].strip()
    dataset_type = request.form['dataset-type'].strip()  # 'Binary' or 'Continuous'

    error = None
    if not sdfile:
        flash("No file was attached.", 'danger')
        return redirect(url_for('toxpro.datasets'))

    if sdfile and not sdfile.filename.rsplit('.', 1)[1].lower() in ['csv', 'sdf']:
        flash("The file must be in SDF or CSV format.", 'danger')
        return redirect(url_for('toxpro.datasets'))

    compound_filename = secure_filename(sdfile.filename)
    user_uploaded_file = os.path.join(current_app.instance_path, compound_filename)
    name = ntpath.basename(user_uploaded_file).split('.')[0]
    sdfile.save(user_uploaded_file)

    try:
        if sdfile.filename.lower().endswith('csv'):
            mols_df = pd.read_csv(user_uploaded_file)

            # Auto-detect default Binary column names only if user left fields blank
            if dataset_type == 'Binary':
                if not compound_id_col and 'CID' in mols_df.columns:
                    compound_id_col = 'CID'
                elif not compound_id_col:
                    compound_id_col = 'CMP_ID'

                if not smiles_col and 'SMILES' in mols_df.columns:
                    smiles_col = 'SMILES'
                elif not smiles_col:
                    smiles_col = 'SMILES'

                if not activity_col and 'Activity_Binary' in mols_df.columns:
                    activity_col = 'Activity_Binary'
                elif not activity_col:
                    activity_col = 'Activity'
            else:
                # Keep original behavior for non-Binary uploads
                activity_col = activity_col or 'Activity'
                compound_id_col = compound_id_col or 'CMP_ID'
                smiles_col = smiles_col or 'SMILES'

            # Early required-column check
            required = [smiles_col, activity_col, compound_id_col]
            missing = [c for c in required if c not in mols_df.columns]
            if missing:
                try:
                    os.remove(user_uploaded_file)
                except Exception:
                    pass
                flash(
                    "The following column(s) were not found in the uploaded CSV: "
                    + ", ".join([f"'{c}'" for c in missing]),
                    'danger'
                )
                return redirect(url_for('toxpro.datasets'))

            if len(mols_df) > 1000:
                raise ValueError(f"The dataset '{sdfile.filename}' contains more than 1000 entries.")

            # Drop rows with missing key fields
            missing_rows = mols_df[
                mols_df[smiles_col].isna() |
                mols_df[activity_col].isna() |
                mols_df[compound_id_col].isna()
            ]
            if not missing_rows.empty:
                flash(f"Warning: {len(missing_rows)} row(s) with missing values were skipped.", 'warning')
                mols_df = mols_df.drop(missing_rows.index)

            mols_df[smiles_col] = mols_df[smiles_col].astype(str)
            mols_df[compound_id_col] = mols_df[compound_id_col].astype(str)

            try:
                PandasTools.AddMoleculeColumnToFrame(mols_df, smilesCol=smiles_col)
                mols_df = mols_df[mols_df.ROMol.notnull()]
            except Exception as e:
                raise ValueError(f"Error processing SMILES: {e}")

        else:  # SDF
            mols_df = PandasTools.LoadSDF(user_uploaded_file)

            # Auto-detect default Binary property names only if user left fields blank
            if dataset_type == 'Binary':
                if not compound_id_col and 'CID' in mols_df.columns:
                    compound_id_col = 'CID'
                elif not compound_id_col:
                    compound_id_col = 'CMP_ID'

                if not smiles_col and 'SMILES' in mols_df.columns:
                    smiles_col = 'SMILES'
                elif not smiles_col:
                    smiles_col = 'SMILES'

                if not activity_col and 'Activity_Binary' in mols_df.columns:
                    activity_col = 'Activity_Binary'
                elif not activity_col:
                    activity_col = 'Activity'
            else:
                # Keep original behavior for non-Binary uploads
                activity_col = activity_col or 'Activity'
                compound_id_col = compound_id_col or 'CMP_ID'
                smiles_col = smiles_col or 'SMILES'

            # Early required-property check
            required = [activity_col, compound_id_col]
            missing = [c for c in required if c not in mols_df.columns]
            if missing:
                try:
                    os.remove(user_uploaded_file)
                except Exception:
                    pass
                flash(
                    "The following column(s) were not found in the uploaded SDF properties: "
                    + ", ".join([f"'{c}'" for c in missing]),
                    'danger'
                )
                return render_template('toxpro/datasets.html', user_datasets=list(current_user.datasets))

            if len(mols_df) > 1000:
                raise ValueError(f"The dataset '{sdfile.filename}' contains more than 1000 entries.")

            # Drop rows with missing key fields
            missing_rows = mols_df[mols_df[activity_col].isna() | mols_df[compound_id_col].isna()]
            if not missing_rows.empty:
                flash(f"Warning: {len(missing_rows)} row(s) with missing values were skipped.", 'warning')
                mols_df = mols_df.drop(missing_rows.index)

            mols_df[compound_id_col] = mols_df[compound_id_col].astype(str)
            try:
                PandasTools.AddMoleculeColumnToFrame(mols_df, smilesCol=smiles_col)
            except Exception as e:
                raise ValueError(f"Error creating molecules from SDF: {e}")

    except Exception as e:
        os.remove(user_uploaded_file)
        flash(f"Failed to process the file: {e}", 'danger')
        return render_template('toxpro/datasets.html', user_datasets=list(current_user.datasets))

    os.remove(user_uploaded_file)

    if mols_df.empty:
        flash("No compounds found in the uploaded file.", 'danger')
        return render_template('toxpro/datasets.html', user_datasets=list(current_user.datasets))

    # Final presence check
    missing_cols = []
    if activity_col not in mols_df.columns:
        missing_cols.append(f"Activity name column '{activity_col}'")
    if compound_id_col not in mols_df.columns:
        missing_cols.append(f"Compound ID column '{compound_id_col}'")
    if smiles_col not in mols_df.columns and sdfile.filename.endswith('.csv'):
        missing_cols.append(f"SMILES column '{smiles_col}'")
    if missing_cols:
        flash("Missing: " + ", ".join(missing_cols), 'danger')
        return render_template('toxpro/datasets.html', user_datasets=list(current_user.datasets))

    # --- Type sanity checks on the observed activity values (before DB insert) ---
    # Coerce to numeric once for checks
    _act = pd.to_numeric(mols_df[activity_col], errors='coerce')
    unique_non_na = sorted(set(_act.dropna().unique().tolist()))

    if dataset_type == 'Binary':
        # Count removals due to non-(0/1)
        before = len(mols_df)
        mols_df = mols_df[_act.isin([0, 1])]
        dropped = before - len(mols_df)
        if dropped > 0:
            flash(f"Note: {dropped} row(s) removed because activity values were not 0/1 for a Binary dataset.", "warning")
        if len(mols_df) == 0:
            flash("All rows were non-binary; please upload as 'Continuous' or correct the Activity column.", "danger")
            return render_template('toxpro/datasets.html', user_datasets=list(current_user.datasets))

    elif dataset_type == 'Continuous':
        # If the data looks binary, warn the user
        looks_binary = all(v in [0, 1] for v in unique_non_na) and len(unique_non_na) <= 2
        if looks_binary:
            flash(
                "The uploaded Activity values appear to be only 0/1. "
                "If you intend classification, consider uploading as 'Binary' next time.",
                "warning"
            )

    # --- Create and store Dataset header ---
    try:
        dataset = Dataset(user_id=current_user.id, dataset_name=name, type=dataset_type)
        db.session.add(dataset)
        db.session.commit()
    except exc.IntegrityError:
        db.session.rollback()
        flash(f"Sorry, there is already a dataset named {name}", 'danger')
        return render_template('toxpro/datasets.html', user_datasets=list(current_user.datasets))

    # --- Persist chemicals ---
    if dataset_type == 'Binary':
        mols_df[activity_col] = pd.to_numeric(mols_df[activity_col], errors='coerce')
        mols_df = mols_df[mols_df[activity_col].notnull()]
        mols_df = mols_df[mols_df[compound_id_col].notnull()]
        mols_df[activity_col] = mols_df[activity_col].astype(float)
        mols_df = mols_df.sort_values(activity_col, ascending=False).drop_duplicates(compound_id_col, keep='first')

        for _, row in mols_df.iterrows():
            mol = row.ROMol
            activity = row[activity_col]
            cmp_id = row[compound_id_col]
            inchi = Chem.MolToInchi(mol)
            if cmp_id and inchi and activity in [0.0, 1.0]:
                dataset.chemicals.append(Chemical(inchi=inchi, dataset_id=dataset.id, activity=activity, compound_id=cmp_id))

    else:  # Continuous
        mols_df[activity_col] = pd.to_numeric(mols_df[activity_col], errors='coerce')
        mols_df = mols_df[mols_df[activity_col].notnull() & mols_df[compound_id_col].notnull()]
        mols_df[activity_col] = mols_df[activity_col].astype(float)
        mols_df[compound_id_col] = mols_df[compound_id_col].astype(str)
        mols_df = mols_df.sort_values(activity_col, ascending=False).drop_duplicates(compound_id_col, keep='first')

        for _, row in mols_df.iterrows():
            mol = row.ROMol
            activity = row[activity_col]
            cmp_id = row[compound_id_col]
            inchi = Chem.MolToInchi(mol)
            if cmp_id and inchi and activity is not None:
                dataset.chemicals.append(Chemical(inchi=inchi, dataset_id=dataset.id, activity=activity, compound_id=cmp_id))

    db.session.add(dataset)
    db.session.commit()

    num_chemicals = len(list(dataset.chemicals))
    flash(f'Uploaded {name} as a new dataset; Added {num_chemicals} chemicals', 'success')
    return redirect(url_for('toxpro.datasets'))


@bp.route('/remove_dataset', methods=['POST'])
@login_required
def remove_dataset():
    dataset_selection = request.form['dataset-selection'].strip()
    do_what_with_dataset = request.form['action']

    if do_what_with_dataset == 'Download dataset as CSV file':
        query_statement = db.session.query(Chemical).join(Dataset,
                                                          Dataset.id == Chemical.dataset_id) \
            .filter(Dataset.dataset_name == dataset_selection) \
            .filter(Dataset.user_id == current_user.id).statement

        df = pd.read_sql(query_statement, db.session.connection())

        # Drop unwanted columns and reorder
        columns_to_keep = ['compound_id', 'inchi', 'activity']
        df = df[[col for col in columns_to_keep if col in df.columns]]  # ensure safe selection

        # Convert to CSV without index
        import io
        mem = io.BytesIO()
        mem.write(df.to_csv(index=False).encode())  # no index, just data
        mem.seek(0)

        return send_file(
            mem,
            as_attachment=True,
            download_name=f"{dataset_selection}.csv",
            mimetype="text/plain",
        )

    # Handle dataset deletion
    dataset = Dataset.query.filter_by(dataset_name=dataset_selection, user_id=current_user.id).first()
    db.session.delete(dataset)
    db.session.commit()
    flash(f"Removed {dataset_selection}", 'danger')

    return redirect(url_for('toxpro.datasets'))


@bp.route('/import_pubchem', methods=['POST'])
@login_required
def import_pubchem():
    """
    import a pubchem data set
    """

    pubchem_aid_string = request.form.get('pubchem_aid', None)

    try:
        pubchem_aid = int(pubchem_aid_string)
    except ValueError:
        pubchem_aid = None

    if pubchem_aid == None:
        flash(f'"{pubchem_aid_string}" is not a valid PubChem AID', 'danger')
        return redirect(url_for('toxpro.datasets'))

    name, reason = pc.get_assay_name(pubchem_aid)
    if reason is not None:
        flash(f'Could not find AID {pubchem_aid}', 'danger')
        return redirect(url_for('toxpro.datasets'))

    current_user.launch_task('add_pubchem_data',
                             f'Importing structure-activity data for AID {pubchem_aid}: {name}',
                             pubchem_aid,
                             current_user.id
                             )
    db.session.commit()
    #flash(
    #f"Task submitted for AID {pubchem_aid}: {name}."
#)

    # if not sdfile:
    #     error = "No SDFile was attached."
    #
    # if sdfile and not sdfile.filename.rsplit('.', 1)[1] in ['sdf']:
    #     error = "The file is not an SDF"
    #
    # if sdfile:
    #     compound_filename = secure_filename(sdfile.filename)
    #
    #     user_uploaded_file = os.path.join(current_app.instance_path, compound_filename)
    #     name = ntpath.basename(user_uploaded_file).split('.')[0]
    #
    #     # create the dataset
    #     try:
    #         dataset = Dataset(user_id=current_user.id, dataset_name=name)
    #         db.session.add(dataset)
    #         db.session.commit()
    #     except exc.IntegrityError:
    #         db.session.rollback()
    #         error = f'Sorry, there is already a dataset named {name}'
    #         flash(error, 'danger')
    #         return redirect(url_for('toxpro.datasets'))
    #
    #     # I think we have to save this in order to use it, not sure if we car read it otherwise
    #     sdfile.save(user_uploaded_file)
    #
    #     mols_df = PandasTools.LoadSDF(user_uploaded_file)
    #     os.remove(user_uploaded_file)
    #
    #     if mols_df.empty:
    #         error = 'No compounds in SDFile'
    #     if activity_col not in mols_df.columns:
    #         error = f'Activity {activity_col} not in SDFile.'
    #     if compound_id_col not in mols_df.columns:
    #         error = f'Compound ID {compound_id_col} not in SDFile.'
    #
    #     if error == None:
    #
    #         # coerce activity column to be
    #         # integer
    #         mols_df[activity_col] = pd.to_numeric(mols_df[activity_col], errors='coerce')
    #         mols_df = mols_df[mols_df[activity_col].notnull()]
    #         mols_df = mols_df[(mols_df[activity_col] == 0) | (mols_df[activity_col] == 1)]
    #         mols_df[activity_col] = mols_df[activity_col].astype(int)
    #
    #         mols_df = mols_df.sort_values(activity_col, ascending=False)
    #         mols_df = mols_df.drop_duplicates(compound_id_col, keep='first')
    #
    #         for i, row in mols_df.iterrows():
    #             mol = row.ROMol
    #             activity = row[activity_col]
    #             cmp_id = row[compound_id_col]
    #             inchi = Chem.MolToInchi(mol)
    #
    #             if cmp_id and inchi and (activity in [1, 0, '1', '0', '1.0', '0.0']):
    #                 chem = Chemical(inchi=inchi, dataset_id=dataset.id, activity=activity, compound_id=cmp_id)
    #                 dataset.chemicals.append(chem)
    #
    #         db.session.add(dataset)
    #         db.session.commit()
    #
    #         num_chemicals = len(list(dataset.chemicals))
    #         flash(f'Uploaded {name} as a new dataset; Added {num_chemicals} chemicals', 'success')
    #         return redirect(url_for('toxpro.datasets'))
    #     else:
    #         db.session.delete(dataset)
    #         db.session.commit()
    #
    # flash(error, 'danger')

    return redirect(url_for('toxpro.datasets'))


@bp.route('/toxdata', methods=['GET'])
@login_required
def toxdata():
    # Support both old flow (direct CURRENT_DATABASES) and new flow (lazy loading function)
    if hasattr(master_db, 'get_current_databases'):
        # New flow: lazy loading function exists
        current_dbs = master_db.get_current_databases()
    else:
        # Old flow: use direct CURRENT_DATABASES variable
        current_dbs = master_db.CURRENT_DATABASES

    return render_template('toxpro/toxdata.html',
                           current_dbs=current_dbs,
                           endpoints=TOXICITY_ENDPOINT_INFO.to_dict('records'))
    
    

@bp.route('/assayProfile', methods=('GET', 'POST'))
@login_required
def assayProfile():
    enable_download_buttons = False  # Disable by default for GET request
    print("Received request for assayProfile.")

    current_user.has_bioprofile = any(d.bioprofile for d in current_user.datasets)
    if request.method == 'GET':
        return render_template('toxpro/assayProfile.html', enable_download_buttons=enable_download_buttons, user_datasets=list(current_user.datasets), user=current_user, label_text="")

    session_dir = SessionManager.get_session_folder()
    session_id = SessionManager.get_or_create_session_id()

    # Create Job record for tracking
    job_id = uuid.uuid4().hex
    job = Job(
        job_id=job_id,
        session_id=session_id,
        job_type='bioprofiler',
        status='queued',
        user_id=current_user.id,
        created_at=datetime.utcnow()
    )
    db.session.add(job)
    db.session.commit()

    # Generate job link for user
    job_link = url_for('toxpro.job_status', job_id=job_id, _external=True)

    # Get uploaded file
    file = request.files.get('dataset-file')
    if not file:
        flash("No file uploaded. Please upload a valid file (.txt or .csv).", 'danger')
        return redirect(url_for('toxpro.assayProfile'))

    # Validate file type
    if not (file.filename.endswith('.txt') or file.filename.endswith('.csv')):
        flash("Invalid file type. Please upload a .txt or .csv file.", 'danger')
        return redirect(url_for('toxpro.assayProfile'))

    filename, file_extension = os.path.splitext(file.filename)
    profile = filename + "_bioprofile"

    # Save the file
    file_path = os.path.join(session_dir, file.filename)
    file.save(file_path)
    print(f"File saved at: {file_path}.")

    # Try to enqueue the background task, fallback to synchronous if Redis unavailable
    try:
        if hasattr(current_app, 'task_queue') and current_app.task_queue:
            # Enqueue the background task
            current_app.task_queue.enqueue(
                'app.tasks.process_bioprofiler_job',
                job_id,
                file_path,
                session_dir
            )

            # Show success message with job link IMMEDIATELY
            flash(f"Your Bioprofiler job has been submitted successfully!", 'success')
            flash(f"Track your job status here: {job_link}", 'info')
            flash(f"This link will remain active for 24 hours. You can bookmark it to check results later.", 'info')

            # Redirect to job status page immediately
            return redirect(job_link)
        else:
            # Redis not available - fall back to synchronous processing
            print("Warning: Redis not available. Processing synchronously...")
            raise ConnectionError("Redis not available")

    except (ConnectionError, Exception) as e:
        print(f"Background task queue unavailable: {e}. Running synchronously...")

        # Fallback to synchronous processing
        from app.tasks import process_bioprofiler_job
        try:
            # Process synchronously
            process_bioprofiler_job(job_id, file_path, session_dir)

            # Show success message
            flash(f"Bioprofiler completed successfully!", 'success')
            flash(f"View your results here: {job_link}", 'info')

            # Redirect to job status page
            return redirect(job_link)

        except Exception as sync_error:
            print(f"Error during synchronous processing: {sync_error}")
            job.status = 'failed'
            job.updated_at = datetime.utcnow()
            db.session.commit()
            flash("An error occurred while processing your file.", 'danger')
            return redirect(url_for('toxpro.assayProfile')) 

import matplotlib
# Use a non-interactive backend
matplotlib.use('Agg')
 

@bp.route('/download_bioprofile', methods=['POST'])
@login_required
def download_bioprofile():

    req_session_id= request.form.get('session_id')  # ✅ Get session ID from request
    print(f"Session ID: {req_session_id}")  # Debug log: session ID
    session_folder = SessionManager.get_folder_by_session_id(req_session_id)
    print(f"Session folder: {session_folder}")  # Debug log: session folder
    do_what = request.form['action']

    if do_what == 'Download Bioprofile':
        # Change file storage to the 'data' folder
        outfile = os.path.join(session_folder, 'bioprofile_long.csv')
        matrix_outfile = os.path.join(session_folder, 'bioprofile_matrix.csv')

        # Generate matrix file
        matrix = bpp.make_matrix(outfile, min_actives=1, outfile=matrix_outfile)

        return send_file(
            matrix_outfile,
            as_attachment=True,
            download_name=f"Initial Bioprofile.csv",
            mimetype="text/plain",
        )

    if do_what == 'Download Heatmap':
        heatmap_outfile = os.path.join(session_folder, 'heatmap.png')

        if os.path.exists(heatmap_outfile):
            print("Heatmap file exists, returning the file.")
            return send_file(
                heatmap_outfile,
                mimetype='image/png',
                as_attachment=True,
                download_name="Heatmap.png"
            )

    if do_what == 'Download Model Metrics':
        metrics_outfile = os.path.join(session_folder, 'RF_metrics.csv')

        if os.path.exists(metrics_outfile):
            return send_file(
                metrics_outfile,
                as_attachment=True,
                download_name="Model Metrics.csv",
                mimetype="text/csv",
            )

    if do_what == 'Download Model Metrics Plot':
        plot_outfile = os.path.join(session_folder, 'RF_metrics_plot.png')

        if os.path.exists(plot_outfile):
            return send_file(
                plot_outfile,
                mimetype='image/png',
                as_attachment=True,
                download_name="Metrics Plot.png",
            )

    if do_what == 'Download Complete Bioprofile':
        filled_matrix = os.path.join(session_folder, 'DataFilled_bioprofile_matrix.csv')

        if os.path.exists(filled_matrix):
            return send_file(
                filled_matrix,
                as_attachment=True,
                download_name="Complete Bioprofile.csv",
                mimetype="text/csv",
            )

@bp.route('/download_database', methods=['POST'])
@login_required
def download_database():
    database_selection = request.form['database-selection'].strip()
    df = master_db.get_raw_table(database_selection)

    # Define columns to drop
    columns_to_exclude = [
        'index', 'CleanedInChI', 'CID_COL', 'Name', 'SMILES', 'Original-Mol-ID', 'No.', 'Dataset-ID',
        'ID', 'Column0', 'ACTIVITY2', 'Column1', 'CAS #', 'DATABASE_NAME'
    ]

    # Drop only if they exist
    df = df.drop(columns=[col for col in columns_to_exclude if col in df.columns], errors='ignore')
    #df = df.drop_duplicates()
    
    # Rename Master-ID to ToxiVerse-ID
    if 'Master-ID' in df.columns:
        df = df.rename(columns={'Master-ID': 'ToxiVerse-ID'})
        
    # Reorder columns: ToxiVerse-ID and Canonical SMILES first
    first_cols = [col for col in ['ToxiVerse-ID', 'CID', 'Canonical SMILES'] if col in df.columns]
    other_cols = [col for col in df.columns if col not in first_cols]
    df = df[first_cols + other_cols]

    # Write to memory buffer
    import io
    mem = io.BytesIO()
    mem.write(df.to_csv(index=False).encode())  # Important: index=False
    mem.seek(0)

    return send_file(
        mem,
        as_attachment=True,
        download_name=f"{database_selection.replace('_curated', '')}.csv",
        mimetype="text/plain",
    )



@bp.route('/example_dataset', methods=['GET'])
def example_file():
    # Get the requested format from query parameters
    file_format = request.args.get('format', '').lower()

    # Define available sample files
    sample_files = {
        'csv': os.path.join(SAMPLE_FILES_DIR, 'sample_csv_dataset.csv'),
        'sdf': os.path.join(SAMPLE_FILES_DIR, 'sample_sdf_dataset.sdf'),
        'prediction_csv': os.path.join(SAMPLE_FILES_DIR, 'prediction_sample_dataset.csv'),
        'prediction_sdf': os.path.join(SAMPLE_FILES_DIR, 'prediction_sample_dataset.sdf')
    }

    # Validate requested format
    if file_format not in sample_files:
        abort(400, description="Invalid file format. Supported formats are 'csv', 'sdf', 'prediction_csv', and 'prediction_sdf'.")

    # Resolve file path
    file_path = sample_files[file_format]
    print(f"Serving sample file: {file_path}")

    # Ensure the file exists
    if not os.path.exists(file_path):
        abort(404, description="Sample file not found.")

    # Define user-friendly download names
    filename_map = {
        'csv': 'QSAR_sample_dataset.csv',
        'sdf': 'QSAR_sample_dataset.sdf',
        'prediction_csv': 'prediction_sample_dataset.csv',
        'prediction_sdf': 'prediction_sample_dataset.sdf'
    }

    return send_file(file_path, as_attachment=True, download_name=filename_map[file_format])


 
@bp.route('/sample', methods=['GET'])
def sample():
  
    # Get the requested format from query parameters
    file_format = request.args.get('format', '').lower()

    # Define available sample files
    sample_files = {
        'csv': os.path.join(SAMPLE_FILES_DIR, 'assay_profile_sample_csv.csv'),
        'txt': os.path.join(SAMPLE_FILES_DIR, 'assay_profile_sample_txt.txt')
    }

     
    # Get the file path for the requested format
    file_path = sample_files[file_format]
    print(file_path)
    # Verify if the file exists
    if not os.path.exists(file_path):
        abort(404, description="Sample file not found.")

    # Send the file as an attachment
    return send_file(file_path, as_attachment=True, download_name=f'Bioprofile_sample_dataset.{file_format}')

# @bp.route('/tutorial', methods=['GET'])
# def tutorial():
  
#     # Get the requested format from query parameters
#     file_format = request.args.get('format', '').lower()

#     # Define available sample files
#     sample_files = {
#         'pdf': os.path.join(SAMPLE_FILES_DIR, 'Tutorial.pdf'),
#     }

#     # Get the file path for the requested format
#     file_path = sample_files[file_format]
#     print(file_path)
#     # Verify if the file exists
#     if not os.path.exists(file_path):
#         abort(404, description="Sample file not found.")

#     # Send the file as an attachment
#     return send_file(file_path, as_attachment=True, download_name=f'Tutorial.{file_format}')

# @bp.route('/data_sou', methods=['GET'])
# def data_sou():
  
#     # Get the requested format from query parameters
#     file_format = request.args.get('format', '').lower()

#     # Define available sample files
#     sample_files = {
#         'docx': os.path.join(SAMPLE_FILES_DIR, 'sample_table.docx'),
#     }
     
#     # Get the file path for the requested format
#     file_path = sample_files[file_format]
#     print(file_path)
#     # Verify if the file exists
#     if not os.path.exists(file_path):
#         abort(404, description="Sample file not found.")

#     # Send the file as an attachment
#     return send_file(file_path, as_attachment=True, download_name=f'sample_table.{file_format}')

@bp.route('/job/<job_id>', methods=['GET'])
def job_status(job_id):
    """
    Display job status and results for Bioprofiler/QSAR jobs
    Accessible via unique job link for 24 hours
    """
    # Find job by job_id
    job = Job.query.filter_by(job_id=job_id).first()
    
    if not job:
        flash("Job not found or expired. Jobs are only accessible for 24 hours.", 'danger')
        return redirect(url_for('toxpro.index'))

    # Check if job is older than 24 hours
    from datetime import timedelta
    if datetime.utcnow() - job.created_at > timedelta(hours=24):
        flash("This job has expired. Jobs are only accessible for 24 hours.", 'warning')
        return redirect(url_for('toxpro.index'))

    # Get session folder path (only for bioprofiler jobs)
    session_folder = None
    if job.job_type == 'bioprofiler':
        session_folder = SessionManager.get_folder_by_session_id(job.session_id)

        # Check if session folder still exists
        if not session_folder or not os.path.exists(session_folder):
            job.status = 'failed'
            db.session.commit()
            flash("Job data not found. It may have been cleaned up.", 'danger')
            return redirect(url_for('toxpro.index'))

    # Get QSAR models if this is a QSAR job
    qsar_models = None
    user_session_id = None
    if job.job_type == 'qsar_build':
        from app.db_models import QSARModel, User
        qsar_models = QSARModel.query.filter_by(user_id=job.user_id).order_by(QSARModel.created.desc()).all()

        # Get the user's session_id for the QSAR predict link
        user = User.query.get(job.user_id)
        if user:
            user_session_id = user.session_id

    # Render job status page
    return render_template('toxpro/job_status.html',
                         job=job,
                         session_id=job.session_id,
                         user_session_id=user_session_id,
                         qsar_models=qsar_models)
