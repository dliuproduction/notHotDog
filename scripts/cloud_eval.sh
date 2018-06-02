gcloud ml-engine jobs submit training object_detection_eval_`date +%s` \
    --job-dir=gs://oxford_pet_dataset/model/train \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.eval \
    --region us-central1 \
    --scale-tier BASIC_GPU \
    -- \
    --checkpoint_dir=gs://oxford_pet_dataset/model/train \
    --eval_dir=gs://oxford_pet_dataset/model/eval \
    --pipeline_config_path=gs://oxford_pet_dataset/model/ssd_mobilenet_v1_pets.config
