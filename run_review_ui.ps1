$ErrorActionPreference = "Stop"

python "$PSScriptRoot\tools\fall_review_ui.py" `
  --runs-dir "$PSScriptRoot\output_runs"
