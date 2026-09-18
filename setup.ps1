# First-time setup wizard for a completely fresh machine. Shows a visible
# checklist window, meant to be run once, on purpose, by someone watching it
# (e.g. handing this app to someone on a brand new PC). Fish Evolution is a
# static HTML/CSS/JS page with no server and no npm dependencies, so the only
# real requirement is a Chromium browser to run it in app mode.

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$appRoot = $PSScriptRoot
$appName = "Fish Evolution"

# ---------- Checklist window ----------

$steps = @(
    "Chrome or Edge browser",
    "Desktop shortcut"
)

$form = New-Object System.Windows.Forms.Form
$form.Text = "$appName Setup"
$form.Size = New-Object System.Drawing.Size(480, 260)
$form.StartPosition = "CenterScreen"
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox = $false
$form.MinimizeBox = $false
try { $form.Icon = [System.Drawing.Icon]::ExtractAssociatedIcon((Join-Path $appRoot "icon.ico")) } catch {}

$title = New-Object System.Windows.Forms.Label
$title.Text = "Setting up $appName"
$title.Font = New-Object System.Drawing.Font("Segoe UI", 14, [System.Drawing.FontStyle]::Bold)
$title.AutoSize = $true
$title.Location = New-Object System.Drawing.Point(20, 18)
$form.Controls.Add($title)

$subtitle = New-Object System.Windows.Forms.Label
$subtitle.Text = "Checking what this computer needs, and installing anything missing."
$subtitle.Font = New-Object System.Drawing.Font("Segoe UI", 9)
$subtitle.ForeColor = [System.Drawing.Color]::DimGray
$subtitle.AutoSize = $true
$subtitle.Location = New-Object System.Drawing.Point(21, 48)
$form.Controls.Add($subtitle)

$statusLabels = @()
$y = 90
foreach ($step in $steps) {
    $lbl = New-Object System.Windows.Forms.Label
    $lbl.Text = "$([char]0x2B1C)   $step"
    $lbl.Font = New-Object System.Drawing.Font("Segoe UI", 10.5)
    $lbl.AutoSize = $false
    $lbl.Location = New-Object System.Drawing.Point(24, $y)
    $lbl.Size = New-Object System.Drawing.Size(430, 22)
    $form.Controls.Add($lbl)

    $detail = New-Object System.Windows.Forms.Label
    $detail.Text = ""
    $detail.Font = New-Object System.Drawing.Font("Segoe UI", 8.5)
    $detail.ForeColor = [System.Drawing.Color]::DimGray
    $detail.AutoSize = $false
    $detail.Location = New-Object System.Drawing.Point(44, ($y + 20))
    $detail.Size = New-Object System.Drawing.Size(410, 18)
    $form.Controls.Add($detail)

    $statusLabels += , @{ Main = $lbl; Detail = $detail }
    $y += 46
}

$progressBar = New-Object System.Windows.Forms.ProgressBar
$progressBar.Style = "Marquee"
$progressBar.MarqueeAnimationSpeed = 30
$progressBar.Location = New-Object System.Drawing.Point(20, ($y + 6))
$progressBar.Size = New-Object System.Drawing.Size(430, 12)
$form.Controls.Add($progressBar)

$actionBtn = New-Object System.Windows.Forms.Button
$actionBtn.Text = "Cancel"
$actionBtn.Size = New-Object System.Drawing.Size(110, 32)
$actionBtn.Location = New-Object System.Drawing.Point(340, ($y + 30))
$script:cancelled = $false
$script:setupComplete = $false
$script:launchNow = $false
$actionBtn.Add_Click({
    if ($script:setupComplete) { $script:launchNow = $true } else { $script:cancelled = $true }
    $form.Close()
})
$form.Controls.Add($actionBtn)

$form.Add_Shown({ $form.Activate() })
$form.Show()
[System.Windows.Forms.Application]::DoEvents()

function Set-Step($index, $status, $detail) {
    $row = $statusLabels[$index]
    $icon = switch ($status) {
        "working" { [char]0x231B }
        "done" { [char]0x2705 }
        "failed" { [char]0x274C }
        default { [char]0x2B1C }
    }
    $row.Main.Text = "$icon   $($steps[$index])"
    $row.Detail.Text = $detail
    $form.Refresh()
    [System.Windows.Forms.Application]::DoEvents()
}

function Show-Result($message, [bool]$isError) {
    $icon = if ($isError) { "Error" } else { "Information" }
    [System.Windows.Forms.MessageBox]::Show($message, "$appName Setup", "OK", $icon) | Out-Null
}

$script:browserPath = $null
$script:fileUrl = $null

try {
    # ---------- 1. Browser ----------
    Set-Step 0 "working" "checking..."
    $chromeCandidates = @(
        "C:\Program Files\Google\Chrome\Application\chrome.exe",
        "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        (Join-Path $env:LOCALAPPDATA "Google\Chrome\Application\chrome.exe")
    )
    $edgeCandidates = @(
        "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        (Join-Path $env:LOCALAPPDATA "Microsoft\Edge\Application\msedge.exe")
    )
    $browser = ($chromeCandidates + $edgeCandidates) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($browser) {
        $browserName = if ($browser -match "Chrome") { "Google Chrome" } else { "Microsoft Edge" }
        Set-Step 0 "done" "found ($browserName)"
        $script:browserPath = $browser
    } else {
        Set-Step 0 "failed" "neither found"
        throw "Neither Google Chrome nor Microsoft Edge was found. $appName needs one of them to run in app mode — install either, then run this setup again. (This is unusual on Windows, which normally includes Edge.)"
    }

    # ---------- 2. Desktop shortcut ----------
    Set-Step 1 "working" "creating..."
    $indexPath = Join-Path $appRoot "index.html"
    $iconPath = Join-Path $appRoot "icon.ico"
    $fileUrl = ([System.Uri]$indexPath).AbsoluteUri
    $script:fileUrl = $fileUrl
    $desktop = [Environment]::GetFolderPath("Desktop")
    $lnkPath = Join-Path $desktop "$appName.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $lnk = $shell.CreateShortcut($lnkPath)
    $lnk.TargetPath = $script:browserPath
    $lnk.Arguments = "--app=$fileUrl"
    $lnk.IconLocation = "$iconPath,0"
    $lnk.WorkingDirectory = $appRoot
    $lnk.Save()
    Set-Step 1 "done" "created on your Desktop"

    $progressBar.Style = "Blocks"
    $progressBar.Value = 100
    $title.Text = "Setup complete!"
    $subtitle.Text = "Everything's ready."
    $actionBtn.Text = "Open $appName"
    $script:setupComplete = $true
    [System.Windows.Forms.Application]::DoEvents()

    while ($form.Visible) {
        Start-Sleep -Milliseconds 100
        [System.Windows.Forms.Application]::DoEvents()
    }

    if ($script:launchNow) {
        Start-Process -FilePath $script:browserPath -ArgumentList "--app=$($script:fileUrl)"
    }
} catch {
    $form.Close()
    if (-not $script:cancelled) {
        Show-Result "Setup didn't finish:`n`n$($_.Exception.Message)" $true
    }
    exit 1
}
