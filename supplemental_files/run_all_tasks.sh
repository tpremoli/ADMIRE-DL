# This script is used to run all the tasks from sample_configs.
# Run this using the following command:
#       bash run_all_tasks.sh

conda activate training_env

echo "Starting all tasks"

# PT tasks
echo "Starting PT tasks"
echo "Starting pt_ax_densenet121"
python cli.py train -c sample_configs/pt_ax_densenet121.yml
echo "Starting pt_ax_densenet201"
python cli.py train -c sample_configs/pt_ax_densenet201.yml
echo "Starting pt_ax_resnet50"
python cli.py train -c sample_configs/pt_ax_resnet50.yml
echo "Starting pt_ax_resnet152"
python cli.py train -c sample_configs/pt_ax_resnet152.yml
echo "Starting pt_ax_vgg16"
python cli.py train -c sample_configs/pt_ax_vgg16.yml
echo "Starting pt_ax_vgg19"
python cli.py train -c sample_configs/pt_ax_vgg19.yml

# TL tasks: Adam
echo "Starting TL tasks"
echo "Starting tl_ax_densenet121_adam"
python cli.py train -c sample_configs/tl_ax_densenet121_adam.yml
echo "Starting tl_ax_densenet201_adam"
python cli.py train -c sample_configs/tl_ax_densenet201_adam.yml
echo "Starting tl_ax_resnet50_adam"
python cli.py train -c sample_configs/tl_ax_resnet50_adam.yml
echo "Starting tl_ax_resnet152_adam"
python cli.py train -c sample_configs/tl_ax_resnet152_adam.yml
echo "Starting tl_ax_vgg16_adam"
python cli.py train -c sample_configs/tl_ax_vgg16_adam.yml
echo "Starting tl_ax_vgg19_adam"
python cli.py train -c sample_configs/tl_ax_vgg19_adam.yml

# TL tasks: SGD
echo "Starting tl_ax_densenet121_sgd"
python cli.py train -c sample_configs/tl_ax_densenet121_sgd.yml
echo "Starting tl_ax_densenet201_sgd"
python cli.py train -c sample_configs/tl_ax_densenet201_sgd.yml
echo "Starting tl_ax_resnet50_sgd"
python cli.py train -c sample_configs/tl_ax_resnet50_sgd.yml
echo "Starting tl_ax_resnet152_sgd"
python cli.py train -c sample_configs/tl_ax_resnet152_sgd.yml
echo "Starting tl_ax_vgg16_sgd"
python cli.py train -c sample_configs/tl_ax_vgg16_sgd.yml
echo "Starting tl_ax_vgg19_sgd"
python cli.py train -c sample_configs/tl_ax_vgg19_sgd.yml

echo "Done with all tasks"