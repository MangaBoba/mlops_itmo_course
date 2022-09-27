from pathlib import Path

from clearml import Dataset, Task, TaskTypes
from config import AppConfig
from data_preparation import main_actions


def main():
    task: Task = Task.init(
        project_name="CoffeeBeans",
        task_name="data preparation",
        task_type=TaskTypes.data_processing,
    )
    clearml_params = {"dataset_id": "dbd9d8ebae474de9a84bb829e60024af"}

    task.connect(clearml_params)

    dataset_path = Dataset.get(**clearml_params).get_local_copy()
    config: AppConfig = AppConfig.parse_raw()
    config.dataset_path = Path(dataset_path)
    main_actions(config=config)
    dataset = Dataset.create(
        dataset_project="prepared_dataset", dataset_name="prepared_dataset"
    )
    dataset.add_files(config.dataset_output_path)
    task.set_parameter("output_dataset_id", dataset.id)
    dataset.upload()
    dataset.finalize()


if __name__ == "__main__":
    main()
