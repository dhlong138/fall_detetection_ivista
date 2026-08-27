$ErrorActionPreference = "Stop"

$configPath = Join-Path $PSScriptRoot "camera_configs\CAM_VP_2.json"
$config = Get-Content -Raw -Path $configPath | ConvertFrom-Json

python "$PSScriptRoot\single_realtime_yolo11x_pose.py" `
  --no-display `
  --publish `
  --publish-fast-async `
  --url $config.url `
  --cam-id $config.cam_id `
  --kafka-bootstrap-servers $config.kafka_bootstrap_servers `
  --model $config.model `
  --device $config.device `
  --process-width $config.process_width `
  --reconnect-delay $config.reconnect_delay `
  --publish-repeat-rate 2 `
  --publish-always-update `
  --publish-object-update-frames $config.publish_object_update_frames `
  --publish-missing-hold-frames $config.publish_missing_hold_frames `
  --publish-result-id-mode $config.publish_result_id_mode
