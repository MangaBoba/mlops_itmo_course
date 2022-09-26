from pathlib import Path
from clearml import Task, TaskTypes, Dataset
from clearml_pipeline.config import AppConfig
from clearml_pipeline.src.data_validation import main_actions


# def main():
task:Task = Task.init(project_name='CoffeeBeans',
                 task_name="data validation", task_type=TaskTypes.data_processing)

clearml_params = {
    "dataset_id":"dbd9d8ebae474de9a84bb829e60024af"
}
task.connect(clearml_params)
dataset_path = Dataset.get(**clearml_params).get_local_copy()
config: AppConfig = AppConfig.parse_raw()
config.dataset_path = Path(dataset_path)
main_actions(config=config)


# if __name__ == "__main__":
#     main()