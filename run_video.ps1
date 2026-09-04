[CmdletBinding()]
param(
    # Default is the sample video already included in this project.
    [string]$VideoPath = (Join-Path $PSScriptRoot "CAMTESTFALLING_rtsp_loop_cache.avi"),

    [string]$Model = (Join-Path $PSScriptRoot "yolo11x-pose.engine"),
    [string]$Device = "cuda",
    [string]$CameraId = "VIDEO_TEST",
    [string]$OutputRoot = (Join-Path $PSScriptRoot "output_runs"),
    [string]$RunName = "",
    [string]$FallConfig = "",
    [double]$ProgressInterval = 5,
    [int]$FallLlmRetryFrames = 30,
    [switch]$NoPreview,
    [switch]$Loop
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $VideoPath -PathType Leaf)) {
    throw "Khong tim thay video: $VideoPath"
}

if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "Khong tim thay model: $Model"
}

if (-not $RunName) {
    $videoName = [System.IO.Path]::GetFileNameWithoutExtension($VideoPath) -replace '[^a-zA-Z0-9_-]', '_'
    $RunName = "${videoName}_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
}
$runDirectory = Join-Path $OutputRoot $RunName
$runIndex = 1
while (Test-Path -LiteralPath $runDirectory) {
    $runDirectory = Join-Path $OutputRoot ("{0}_{1:D2}" -f $RunName, $runIndex)
    $runIndex++
}
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null

# One pass is the safe default. Add -Loop to replay the video continuously.
$arguments = @(
    "$PSScriptRoot\single_realtime_yolo11x_pose.py",
    "--video", $VideoPath,
    "--model", $Model,
    "--device", $Device,
    "--cam-id", $CameraId,
    "--progress-interval", $ProgressInterval,
    "--fall-llm-retry-frames", $FallLlmRetryFrames,
    "--save-video-dir", $runDirectory,
    "--fall-crops-dir", $runDirectory,
    "--no-publish-sample-resource",
    # A confirmed fall is the only Kafka message. The LLM-confirmed frame is
    # saved/uploaded by the runtime and attached to that alert.
    "--publish",
    "--publish-fall-only",
    "--fall-llm-verify"
)

if ($NoPreview) {
    $arguments += "--no-display"
} else {
    $arguments += "--display"
}

if ($Loop) {
    $arguments += "--loop-video"
} else {
    $arguments += @("--no-loop-video", "--video-loop-count", "1")
}

if ($FallConfig) {
    if (-not (Test-Path -LiteralPath $FallConfig -PathType Leaf)) {
        throw "Khong tim thay fall config: $FallConfig"
    }
    $arguments += @("--settings-file", $FallConfig)
}

Write-Host "Dang chay video: $VideoPath"
Write-Host "Thu muc ket qua: $runDirectory"
Write-Host "Chi gui canh bao nga da duoc LLM xac nhan, kem frame anh."
Write-Host "Nhan q hoac Esc trong cua so preview de dung."
& python @arguments
exit $LASTEXITCODE
