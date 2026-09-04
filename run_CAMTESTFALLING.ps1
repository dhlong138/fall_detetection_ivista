$ErrorActionPreference = "Stop"

$configPath = Join-Path $PSScriptRoot "camera_configs\CAMTESTFALLING.json"
$config = Get-Content -Raw -Path $configPath | ConvertFrom-Json

python "$PSScriptRoot\scripts\loop_realtime_yolo11x_pose.py" `
  --display `
  --publish `
  --publish-fast-async `
  --publish-always-update `
  --publish-repeat-rate 2 `
  --url $config.url `
  --cam-id $config.cam_id `
  --kafka-bootstrap-servers $config.kafka_bootstrap_servers `
  --model $config.model `
  --device $config.device `
  --reconnect-delay $config.reconnect_delay `
  --publish-object-update-frames $config.publish_object_update_frames `
  --publish-missing-hold-frames $config.publish_missing_hold_frames `
  --publish-sample-strict
