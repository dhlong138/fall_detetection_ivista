[CmdletBinding()]
param(
    [string]$VideoPath = "C:\Users\admin\Downloads\video nga\001tong_hop_nga_camVP.mp4",
    [double]$ProgressInterval = 10
)

$ErrorActionPreference = "Stop"
$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "Lan 1/2: cau hinh mac dinh"
& "$PSScriptRoot\run_video.ps1" `
  -VideoPath $VideoPath `
  -RunName "comparison_default_$runStamp" `
  -ProgressInterval $ProgressInterval `
  -NoPreview
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Lan 2/2: cau hinh nhay cho nga xuoi camera"
& "$PSScriptRoot\run_video.ps1" `
  -VideoPath $VideoPath `
  -RunName "comparison_front_camera_$runStamp" `
  -FallConfig "$PSScriptRoot\fall_configs\front_camera_sensitive.json" `
  -ProgressInterval $ProgressInterval `
  -NoPreview
exit $LASTEXITCODE
