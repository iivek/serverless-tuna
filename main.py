"""
Even if the process where the trial is running is killed for some reason, you can restart from
previous saved checkpoint using heartbeat.
"""

import os
import tempfile

import optuna
from optuna.artifacts import download_artifact
from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import upload_artifact
from optuna.storages import RetryFailedTrialCallback


DIR = os.getcwd()
CHECKPOINT_DIR = "pytorch_checkpoint"

base_path = "./artifacts"
os.makedirs(base_path, exist_ok=True)
artifact_store = FileSystemArtifactStore(base_path=base_path)


def objective(trial):
    # Optimizing a quadratic function
    x = trial.suggest_float("x", -10, 10)
    objective_value = (x - 3) ** 2
    # trial.report(objective_value, 0)
    study.get_trials()

    # Recovery attempt
    artifact_id = None
    retry_history = RetryFailedTrialCallback.retry_history(trial)
    for trial_number in reversed(retry_history):
        artifact_id = trial.study.trials[trial_number].user_attrs.get("artifact_id")
        if artifact_id is not None:
            retry_trial_number = trial_number
            break

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")

        # Do we have an artifact to recover from?
        if artifact_id is not None:
            download_artifact(
                artifact_store=artifact_store,
                file_path=checkpoint_path,
                artifact_id=artifact_id,
            )
            print(
                f"Loading checkpoint from trial {retry_trial_number}, epoch {checkpoint}."
            )
            # Call your recovery routines here

            # Simulating checkpointing
            checkpoint_data = {"x": x, "objective_value": objective_value}
            with open(checkpoint_path, "w") as f:
                f.write(str(checkpoint_data))

            artifact_id = upload_artifact(
                artifact_store=artifact_store,
                file_path=checkpoint_path,
                study_or_trial=trial,
            )
            trial.set_user_attr("artifact_id", artifact_id)

        return objective_value


if __name__ == "__main__":
    storage = optuna.storages.RDBStorage(
        "sqlite:///example.db",
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(),
    )
    study = optuna.create_study(
        storage=storage,
        study_name="pytorch_checkpoint",
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100, timeout=600)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    optuna.visualization.plot_intermediate_values(study).show()