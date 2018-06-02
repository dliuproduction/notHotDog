gcloud ml-engine jobs submit training object_detection_`date +%s` \
    --job-dir=gs://oxford_pet_dataset/model/train \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,dist/tensorflow-1.4.0-cp27-cp27mu-linux_x86_64.whl \
    --module-name object_detection.train \
    --region us-central1 \
    --config object_detection/samples/cloud/cloud.yml\
    -- \
    --train_dir=gs://oxford_pet_dataset/model/train \
    --pipeline_config_path=gs://oxford_pet_dataset/model/ssd_mobilenet_v1_pets_cloud.config
