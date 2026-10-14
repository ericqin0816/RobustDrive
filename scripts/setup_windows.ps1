$ErrorActionPreference = 'Stop'
$rdProject = Split-Path -Parent $PSScriptRoot
$rdPython = Join-Path $rdProject '.venv\Scripts\pythonw.exe'
$rdSimulator = Join-Path $rdProject 'simulator\beta_simulator_windows\beta_simulator.exe'
if (-not (Test-Path -LiteralPath $rdPython)) { throw 'Install the project virtual environment first.' }
if (-not (Test-Path -LiteralPath $rdSimulator)) { throw 'Extract the official Windows simulator into simulator first.' }

$rdRecording = Join-Path $rdProject 'data\raw'
New-Item -ItemType Directory -Path $rdRecording -Force | Out-Null
$rdDocuments = [Environment]::GetFolderPath('MyDocuments')
$rdFriendlyFolder = Join-Path $rdDocuments 'RobustDrive Recordings'
if (Test-Path -LiteralPath $rdFriendlyFolder) {
    $rdExisting = Get-Item -LiteralPath $rdFriendlyFolder
    if ($rdExisting.LinkType -ne 'Junction' -or $rdExisting.Target -ne $rdRecording) {
        throw "A different folder already exists at $rdFriendlyFolder; it was left unchanged."
    }
} else {
    New-Item -ItemType Junction -Path $rdFriendlyFolder -Target $rdRecording | Out-Null
}

$rdDesktop = [Environment]::GetFolderPath('Desktop')
$rdShortcutPath = Join-Path $rdDesktop 'Play RobustDrive.lnk'
$rdShell = New-Object -ComObject WScript.Shell
$rdShortcut = $rdShell.CreateShortcut($rdShortcutPath)
$rdShortcut.TargetPath = $rdPython
$rdShortcut.Arguments = '"' + (Join-Path $rdProject 'scripts\play.py') + '"'
$rdShortcut.WorkingDirectory = $rdProject
$rdShortcut.IconLocation = $rdSimulator + ',0'
$rdShortcut.WindowStyle = 1
$rdShortcut.Description = 'Play and record; train your steering models automatically after closing the game.'
$rdShortcut.Save()
Write-Output "Desktop shortcut: $rdShortcutPath"
Write-Output "In-game recording folder: $rdFriendlyFolder"
