# this module is outlined
# here: https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-xxii-background-jobs
from app import create_app
from app.db_models import db, Dataset, Chemical, QSARModel, Task, CVResults, Bioprofile, KDNNPrediction
from app.curator.curator import Curator
from app import chem_io
from app import machine_learning as ml
from app import pubchem as pc
from app import bioprofile as bp
from app import make_new_predictions as knn

import sys, time, pickle, codecs
import datetime
from datetime import timezone
from rq import get_current_job
import pandas as pd
import logging
import re
import os
import shutil

app = create_app()
app.app_context().push()

# Logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_qsar(user_id, dataset_name, descriptors, algorithm, type):
    """Build a QSAR model as a background job."""
    job = get_current_job()
    job.meta['progress'] = 'Creating Features...'
    job.save_meta()

    query_statement = db.session.query(Chemical).join(
        Dataset, Dataset.id == Chemical.dataset_id
    ).filter(
        Dataset.dataset_name == dataset_name,
        Dataset.user_id == user_id
    ).statement
    df = pd.read_sql(query_statement, db.session.connection())

    dataset = Dataset.query.filter_by(dataset_name=dataset_name, user_id=user_id).first()

    # Remove existing model/results for same (user, dataset, algo, desc, type)
    qsar_model = QSARModel.query.filter_by(
        user_id=user_id, algorithm=algorithm, descriptors=descriptors, type=type, dataset_id=dataset.id
    ).first()
    if qsar_model:
        cv_results = CVResults.query.filter_by(qsar_model_id=qsar_model.id).first()
        if cv_results:
            db.session.delete(cv_results)
        db.session.delete(qsar_model)

    # Create descriptors
    X = chem_io.get_desc(df, descriptors)
    y = df['activity']
    y.index = df['compound_id']
    y = y.loc[X.index]

    scale = True if descriptors == 'RDKit' else False

    # Train
    job.meta['progress'] = 'Training Model...'
    job.save_meta()
    if type == 'Classification':
        model, cv_preds, train_stats = ml.build_qsar_model(X, y, algorithm, scale=scale)
    elif type == 'Regression':
        model, cv_preds, train_stats = ml.build_qsar_model_regression(X, y, algorithm, scale=scale)
    else:
        raise ValueError(f"Unknown model type: {type}")

    qsar_model = QSARModel(
        user_id=user_id,
        name=f'{dataset_name}-{descriptors}-{algorithm}-{type}',
        algorithm=algorithm,
        descriptors=descriptors,
        type=type,
        dataset_id=dataset.id,
        sklearn_model=model
    )
    db.session.add(qsar_model)
    db.session.commit()

    cv_results = CVResults(
        qsar_model_id=qsar_model.id,
        accuracy=train_stats['ACC'],
        f1_score=train_stats['F1-Score'],
        area_under_roc=train_stats['AUC'],
        cohens_kappa=train_stats["Cohen's Kappa"],
        # matthews_correlation=train_stats['MCC'],
        precision=train_stats['Precision'],
        recall=train_stats['Recall'],
        specificity=train_stats['Specificity'],
        correct_classification_rate=train_stats['CCR'],
        r2_score=train_stats['R2-score'],
        max_error=train_stats['Max-error'],
        mean_squared_error=train_stats['Mean-squared-error'],
        mean_absolute_percentage_error=train_stats['Mean-absolute-percentage-error'],
        pinball_score=train_stats['D2-pinball-score']
    )
    db.session.add(cv_results)
    db.session.commit()

    # Mark task complete (and COMMIT)
    job.meta['progress'] = 'Complete'
    job.save_meta()
    task = Task.query.get(job.get_id())
    task.complete = True
    task.time_completed = datetime.datetime.now(timezone.utc)
    db.session.commit()


def curate_chems(user_id, dataset_name, duplicate_selection, replace=False):
    """Curate a set of chemicals."""
    job = get_current_job()
    job.meta['progress'] = 'Curating...'
    job.save_meta()
    logger.info(f"Starting curation for user_id: {user_id}, dataset: {dataset_name}")

    query_statement = db.session.query(Chemical).join(
        Dataset, Dataset.id == Chemical.dataset_id
    ).filter(
        Dataset.dataset_name == dataset_name,
        Dataset.user_id == user_id
    ).statement
    df = pd.read_sql(query_statement, db.session.connection())
    dataset = Dataset.query.filter_by(dataset_name=dataset_name, user_id=user_id).first()

    curator = Curator(df)
    curator.curate(duplicates=duplicate_selection)
    new_df = curator.new_df.copy()

    if not replace:
        logger.debug("Create new curated dataset…")
        job.meta['progress'] = 'Adding dataset...'
        job.save_meta()

        new_dataset = Dataset(dataset_name=dataset.dataset_name + '_curated', user_id=user_id)
        db.session.add(new_dataset)
        db.session.commit()

        for _, row in new_df.iterrows():
            chem = Chemical(
                inchi=row['inchi'],
                dataset_id=new_dataset.id,  # <-- new dataset id (fix)
                activity=row['activity'],
                compound_id=row['compound_id']
            )
            new_dataset.chemicals.append(chem)

        db.session.add(new_dataset)
        db.session.commit()
    else:
        logger.debug("Replace existing dataset…")
        job.meta['progress'] = 'Replacing dataset...'
        job.save_meta()

        # remove all previous chemicals
        Chemical.query.filter_by(dataset_id=dataset.id).delete()
        db.session.commit()

        for _, row in new_df.iterrows():
            chem = Chemical(
                inchi=row['inchi'],
                dataset_id=dataset.id,
                activity=row['activity'],
                compound_id=row['compound_id']
            )
            dataset.chemicals.append(chem)

        db.session.add(dataset)
        db.session.commit()

    # Mark task complete (and COMMIT)
    logger.info("Curating task completed; marking complete")
    job.meta['progress'] = 'Complete'
    job.save_meta()
    task = Task.query.get(job.get_id())
    task.complete = True
    task.time_completed = datetime.datetime.now(timezone.utc)
    db.session.commit()


def add_pubchem_data(aid, user_id):
    """Import PubChem AID data and store as a dataset."""
    job = get_current_job()
    job.meta['progress'] = 'Importing data...'
    job.save_meta()

    df, fail_reason = pc.import_pubchem_aid(aid)
    if fail_reason is not None:
        job.meta['progress'] = f'Failed: {fail_reason}'
        job.save_meta()
        return

    job.meta['progress'] = 'Adding data...'
    job.save_meta()

    # delete any existing dataset with same name for this user
    Dataset.query.filter_by(user_id=user_id, dataset_name=f'AID_{aid}').delete()
    db.session.commit()

    dataset = Dataset(dataset_name=f'AID_{aid}', user_id=user_id)
    db.session.add(dataset)
    db.session.commit()

    for _, row in df.iterrows():
        chem = Chemical(
            inchi=row['inchi'],
            dataset_id=dataset.id,
            activity=row['activity'],
            compound_id=row['compound_id']
        )
        dataset.chemicals.append(chem)

    db.session.add(dataset)
    db.session.commit()

    # Mark task complete (and COMMIT)
    job.meta['progress'] = 'Complete'
    job.save_meta()
    task = Task.query.get(job.get_id())
    task.complete = True
    task.time_completed = datetime.datetime.now(timezone.utc)
    db.session.commit()


def build_bioprofile_OG(user_id, dataset_name, dataset_type):
    job = get_current_job()
    job.meta['progress'] = 'Getting bioassays from PubChem...'
    job.save_meta()

    dataset = Dataset.query.filter_by(dataset_name=dataset_name, user_id=user_id).first()

    query_statement = db.session.query(Chemical).join(
        Dataset, Dataset.id == Chemical.dataset_id
    ).filter(
        Dataset.dataset_name == dataset_name,
        Dataset.user_id == user_id
    ).statement
    df = pd.read_sql(query_statement, db.session.connection())

    existing_bioprofile = Bioprofile.query.filter_by(
        user_id=user_id, name=f'{dataset_name}-bioprofile', dataset_id=dataset.id
    ).first()
    if existing_bioprofile:
        db.session.delete(existing_bioprofile)

    df['compound_id'] = df['compound_id'].dropna()
    df = df[df['compound_id'] != 'nan']
    df['compound_id'] = pd.to_numeric(df['compound_id'])

    identifier_list = df['compound_id'].astype(float).astype(int).to_list()
    bp.generate_bioprofile_new(identifier_list)

    job.meta['progress'] = 'Bioassays gotten. Making matrices...'
    job.save_meta()

    outfile = Config.BIOASSAY_OUT  # assuming this exists in your Config
    profile = bp.make_bioprofile(df, preprofile)
    profile_matrix = bp.make_bioprofile_matrix_new(profile, min_actives=10)

    job.meta['progress'] = 'Bioprofile generated. Making Heatmap'
    job.save_meta()

    supervised_profile = bp.render_bioprofile(profile, dataset_type)
    pca = bp.get_pca(supervised_profile, profile)
    heatmap = bp.get_heat_map(profile_matrix)
    active_assay_table = bp.get_active_bioassays(pca)

    bioprofile = Bioprofile(
        user_id=user_id,
        name=f'{dataset_name}-bioprofile',
        bioprofile=pickle.dumps(profile_matrix),
        pca=pickle.dumps(pca),
        heatmap=pickle.dumps(heatmap),
        dataset_id=dataset.id,
        active_assay_table=pickle.dumps(active_assay_table)
    )
    db.session.add(bioprofile)
    db.session.commit()

    job.meta['progress'] = 'Complete'
    job.save_meta()
    task = Task.query.get(job.get_id())
    task.complete = True
    task.time_completed = datetime.datetime.now(timezone.utc)
    db.session.commit()


def build_bioprofile(user_id, dataset_name, dataset_type):
    job = get_current_job()
    job.meta['progress'] = 'Getting bioassays from PubChem...'
    job.save_meta()

    # Fetch dataset
    dataset = Dataset.query.filter_by(dataset_name=dataset_name, user_id=user_id).first()
    print("job info" + job.description)

    match = re.search(r"app\.tasks\.build_bioprofile\(\d+, '(.*?)',", job.description)
    if not match:
        raise ValueError("Dataset path not found in job description.")
    dataset_path = match.group(1)
    print(f"Extracted Dataset Path: {dataset_path}")

    # Load the dataset into a DataFrame
    df = pd.read_csv(dataset_path)
    target_compounds = []

    with open(dataset_path, "r") as csv_file:
        for line in csv_file:
            target_compounds.append(line.strip())

    # Call bioprofile to fetch bioassays and save them to CSV
    outfile = 'bioprofile_long.csv'
    bp.bioprofile(target_compounds, chunk=100, outfile=outfile)

    job.meta['progress'] = 'Bioassays gotten. Making matrices...'
    job.save_meta()

    # Generate matrix using the bioprofile CSV file
    matrix = bp.make_matrix(outfile, min_actives=10, outfile='bioprofile_matrix.csv')

    job.meta['progress'] = 'Bioprofile generated. Making Heatmap'
    job.save_meta()

    # Generate heatmap from the matrix
    bp.get_heat_map(matrix, output_file='heatmap.png')

    bioprofile_obj = Bioprofile(
        user_id=user_id,
        name=f'{dataset_name}-bioprofile',
        bioprofile=pickle.dumps(matrix),
        heatmap=pickle.dumps('heatmap.png'),
        dataset_id=dataset.id
    )
    db.session.add(bioprofile_obj)
    db.session.commit()

    job.meta['progress'] = 'Complete'
    job.save_meta()
    task = Task.query.get(job.get_id())
    task.complete = True
    task.time_completed = datetime.datetime.now(timezone.utc)
    db.session.commit()


def kdnn_predict(user_id, user_uploaded_dataframe, df_name, identifier_column):
    job = get_current_job()
    job.meta['progress'] = 'Predicting on kDNN...'
    job.save_meta()

    predicted_values, score, curve = knn.make_knn_prediction(
        dataframe_name=df_name,
        dataframe=user_uploaded_dataframe,
        identifier_predict=identifier_column
    )

    kdnn_prediction = KDNNPrediction(
        user_id=user_id,
        name=f'{df_name}-kDNN-Prediction',
        prediction=pickle.dumps(predicted_values),
        auc_score=score,
        auc_curve=pickle.dumps(curve)
    )
    db.session.add(kdnn_prediction)
    db.session.commit()

    # Mark task complete (and COMMIT)
    job.meta['progress'] = 'Complete'
    job.save_meta()
    task = Task.query.get(job.get_id())
    task.complete = True
    task.time_completed = datetime.datetime.now(timezone.utc)
    db.session.commit()


def example(seconds):
    job = get_current_job()
    print('Starting task')
    for i in range(seconds):
        job.meta['progress'] = 100.0 * i / seconds
        job.save_meta()
        print(i)
        time.sleep(1)
    job.meta['progress'] = 100
    job.save_meta()
    print('Task completed')


def cleanup_old_sessions(expiry_hours=24):
    """
    Deletes session folders older than the given expiry time (default: 24 hours).
    Expects TMP_FILES_DIR to exist; if not, safely no-op.
    """
    TMP_FILES_DIR = os.path.join(app.instance_path, "tmp_sessions")
    if not os.path.isdir(TMP_FILES_DIR):
        return "No session directory found; nothing to clean."

    now = time.time()
    expiry_threshold = now - (expiry_hours * 3600)  # hours -> seconds

    for folder in os.listdir(TMP_FILES_DIR):
        folder_path = os.path.join(TMP_FILES_DIR, folder)
        if folder.startswith("session_") and os.path.isdir(folder_path):
            folder_age = os.stat(folder_path).st_mtime  # Last modified time
            if folder_age < expiry_threshold:
                shutil.rmtree(folder_path, ignore_errors=True)
                print(f"🗑 Deleted old session folder: {folder_path}")

    return "Session cleanup completed"


if __name__ == '__main__':
    curate_chems(1, "AMES_curated", duplicate_selection="average", replace=False)
