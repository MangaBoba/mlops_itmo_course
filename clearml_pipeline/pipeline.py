from clearml import PipelineController

pipe = PipelineController(
    name="Training pipeline", project="CoffeeBeans", version="0.0.1"
)
pipe.set_default_execution_queue("default")

pipe.add_step(
    name="validation_data",
    # parents=['preparation_data', ],
    base_task_project="CoffeeBeans",
    base_task_name="data validation",
    parameter_override={"General/dataset_id": "dbd9d8ebae474de9a84bb829e60024af"},
)

pipe.add_step(
    name="preparation_data",
    parents=["validation_data"],
    base_task_project="CoffeeBeans",
    base_task_name="data preparation",
    parameter_override={"General/dataset_id": "dbd9d8ebae474de9a84bb829e60024af"},
)

pipe.add_step(
    name="training_step",
    parents=["preparation_data"],
    base_task_project="CoffeeBeans",
    base_task_name="training",
    parameter_override={
        "General/dataset_id": "${preparation_data.parameters.General/output_dataset_id}"
    },
)

# pipe.add_step(
#     name='testing_step',
#     parents=['training_step'],
#     base_task_project='CoffeeBeans',
#     base_task_name='testing',
#     parameter_override={
#         'General/dataset_id': "${preparation_data.parameters.General/output_dataset_id}"},
# )

pipe.start_locally(run_pipeline_steps_locally=True)

print("done")
