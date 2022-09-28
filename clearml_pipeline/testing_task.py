from pathlib import Path

from clearml import Dataset, Task, TaskTypes
from config import AppConfig
from training import test as main_actions


def main():
    task: Task = Task.init(
        project_name="CoffeeBeans", task_name="testing", task_type=TaskTypes.testing
    )

    clearml_params = {"dataset_id": "dbd9d8ebae474de9a84bb829e60024af"}
    task.connect(clearml_params)
    dataset_path = Dataset.get(clearml_params["dataset_id"]).get_local_copy()
    config: AppConfig = AppConfig.parse_raw()
    config.dataset_path = Path(dataset_path)
    # print(config.dataset_path)
    # model = models.resnet18()
    # model.fc = torch.nn.Linear(in_features=512, out_features=4, bias=True)
    # try:
    #     task = Task.get_task(task_id='training_step')
    #     model_pt = task.artifacts["model"].get()
    #     model.load_state_dict(model_pt)
    #     print(model_pt)
    # except:
    #     pass
    prev_task = Task.get_task(project_name="CoffeeBeans", task_id="training")
    model_snap = prev_task.models["output"][-1]
    model = model_snap.get_local_copy()
    main_actions(model, config=config)


if __name__ == "__main__":
    main()
