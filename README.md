# Run on CCV (03/06)

Clone the branch `testing` of the repository.

```
git clone -b testing https://github.com/BatsResearch/co-training-clip.git
```

Within the new folder create three more folders:
- `mkdir pseudolabels`
- `mkdir logs`
- `mkdir trained_prompts`
- `mkdir evaluation`

One level up the repository's folder, create a new environment named zsl.

```
python3 -m venv zsl
```

On CCV, install the required library running with `sbatch` the following script. Note that you have to correctly specify the path of the environment when you activate it.

```bash
#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --output=logs/setup_env.out
#SBATCH --ntasks=1
#SBATCH -t 03:00:00

module load cuda/11.1.1

pwd
source ../zsl/bin/activate

bash setup.sh
```

At this point, you should be ready to launch jobs!

Place yourself in the repository's folder and execute scripts in the `scripts/` folder using sbatch. Note that before launching any script you should double-check that 
1. The path to your environment and files is correct.
2. The port for accelerate is different for each job you run simultaneously
3. Check the configuration file and change the `DATASET_DIR` accordingly




