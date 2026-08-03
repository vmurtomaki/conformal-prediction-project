# scripts/generate_context.ps1
# Requires PowerShell 5.1

$ScriptDir = $PSScriptRoot
$RootDir = (Get-Item $ScriptDir).Parent.FullName
$OutputFile = Join-Path $ScriptDir "llm_context.txt"
$Output = [System.Text.StringBuilder]::new()

[void]$Output.AppendLine("<project_tree>")
[void]$Output.AppendLine((Get-Item $RootDir).Name)

$Script:FilesToInclude = [System.Collections.Generic.List[System.IO.FileInfo]]::new()

function ShouldIncludeContent($File) {
    $ExcludeExtensions = @(
        '.png', '.jpg', '.jpeg', '.webp', '.gif', '.pdf', '.zip', '.tar', '.gz', 
        '.lock', '.faiss', '.pt', '.pth', '.tensors', '.index', '.pyc', '.pyd', '.csv'
    )
    $ExcludeNames = @(
        '.gitignore', '.python-version', 'uv.lock', 'llm_context.txt', 
        'generate_context.ps1', 'CACHEDIR.TAG'
    )

    if ($ExcludeExtensions -contains $File.Extension.ToLower()) { return $false }
    if ($ExcludeNames -contains $File.Name) { return $false }
    return $true
}

function Get-Tree($CurrentDir, $Indent) {
    # Tailored exclusions for Conformal_Prediction_Project
    $ExcludeDirs = @('.git', '.venv', '__pycache__', 'data', '.pytest_cache')

    $Dirs = Get-ChildItem -Path $CurrentDir.FullName -Directory | 
            Where-Object { $_.Name -notin $ExcludeDirs }
    $Files = Get-ChildItem -Path $CurrentDir.FullName -File

    for ($i = 0; $i -lt $Dirs.Count; $i++) {
        $IsLast = ($i -eq $Dirs.Count - 1) -and ($Files.Count -eq 0)
        $Marker = if ($IsLast) { "\-- " } else { "+-- " }
        [void]$Output.AppendLine(($Indent + $Marker + $Dirs[$i].Name))
        
        $NextIndent = if ($IsLast) { $Indent + "    " } else { $Indent + "|   " }
        Get-Tree $Dirs[$i] $NextIndent
    }

    for ($i = 0; $i -lt $Files.Count; $i++) {
        $IsLast = $i -eq ($Files.Count - 1)
        $Marker = if ($IsLast) { "\-- " } else { "+-- " }
        [void]$Output.AppendLine(($Indent + $Marker + $Files[$i].Name))

        if (ShouldIncludeContent $Files[$i]) {
            $Script:FilesToInclude.Add($Files[$i])
        }
    }
}

Write-Host "Scanning Conformal_Prediction_Project structure..." -ForegroundColor Cyan
Get-Tree (Get-Item $RootDir) ""
[void]$Output.AppendLine("</project_tree>")
[void]$Output.AppendLine("")

$LargeFilesFound = $false
foreach ($File in $Script:FilesToInclude) {
    $LineCount = 0
    Get-Content $File.FullName -ReadCount 1000 | ForEach-Object { $LineCount += $_.Count }
    if ($LineCount -gt 500) {
        Write-Warning "File exceeds 500 lines: $($File.Name) ($LineCount lines)"
        $LargeFilesFound = $true
    }
}

if ($LargeFilesFound) {
    Write-Host ""
    Read-Host -Prompt "WARNING: Large files found. Press [Enter] to continue, or [Ctrl+C] to abort"
}

Write-Host "Compiling context payload..." -ForegroundColor Cyan
foreach ($File in $Script:FilesToInclude) {
    $RelativePath = $File.FullName.Replace($RootDir, "").TrimStart("\").TrimStart("/").Replace("\", "/")
    [void]$Output.AppendLine("<file path=`"$RelativePath`">")
    try {
        $Content = (Get-Content -Path $File.FullName -Raw -Encoding UTF8).Trim()
        [void]$Output.AppendLine($Content)
    } catch {
        [void]$Output.AppendLine("[error reading file]")
    }
    [void]$Output.AppendLine("</file>")
}

$FinalString = $Output.ToString()
$FinalString | Out-File -FilePath $OutputFile -Encoding utf8
$FinalString | Set-Clipboard

Write-Host "Success! Context copied and saved." -ForegroundColor Green