from pathlib import Path

PROJECT_PATH = Path(__file__).parent.parent

CONFIGS_PATH = PROJECT_PATH / 'configs'
DATA_PATH = PROJECT_PATH / 'data'
EXPERIMENTS_PATH = PROJECT_PATH / 'experiments'
PL_LOGS_PATH = PROJECT_PATH / 'lightning_logs'
ANNOTATIONS_PATH = DATA_PATH / 'annotations.tsv'
