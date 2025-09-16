INPUT_DS="/teamspace/studios/this_studio/ampa_ds"
OUTPUT_DIR="/teamspace/studios/this_studio/ampa_flat"
SUBJECTS_DIR="/teamspace/studios/this_studio/freesurfer_subjects"
DS_NAME="ampa_flat"

export FREESURFER_HOME=/usr/local/freesurfer/8.1.0
source $FREESURFER_HOME/SetUpFreeSurfer.sh

uv run process_bids_to_flatmaps.py \
  --bids_dir $INPUT_DS \
  --output_dir $OUTPUT_DIR \
  --subjects_dir $SUBJECTS_DIR \
  --task rest \
  --dataset_name $DS_NAME \
  --dry_run