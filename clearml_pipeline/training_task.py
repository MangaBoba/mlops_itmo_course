from pathlib import Path

from clearml import Dataset, Task, TaskTypes
from config import AppConfig
from training import main_actions


def main():
    task: Task = Task.init(
        project_name="CoffeeBeans", task_name="training", task_type=TaskTypes.training
    )

    clearml_params = {"dataset_id": "dbd9d8ebae474de9a84bb829e60024af"}
    task.connect(clearml_params)
    dataset_path = Dataset.get(clearml_params["dataset_id"]).get_local_copy()
    config: AppConfig = AppConfig.parse_raw()
    config.dataset_path = Path(dataset_path)
    print(config.dataset_path)
    model = main_actions(config=config)
    return model
    # task.upload_artifact('beans_classifier', artifact_object=model, auto_pickle = True)


if __name__ == "__main__":
    main()
