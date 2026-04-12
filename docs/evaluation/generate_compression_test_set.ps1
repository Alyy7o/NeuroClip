param(
    [Parameter(Mandatory=$true)]
    [string]$InputDir,

    [Parameter(Mandatory=$true)]
    [string]$OutputDir
)

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    Write-Error "ffmpeg not found in PATH. Install ffmpeg and try again."
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$profiles = @(
    @{ Id='CMP_TINY_01'; Duration=30;  Scale='640:360';  VBitrate='500k';  ABitrate='64k'  },
    @{ Id='CMP_SMALL_02'; Duration=60;  Scale='854:480';  VBitrate='800k';  ABitrate='96k'  },
    @{ Id='CMP_MEDIUM_03'; Duration=180; Scale='1280:720'; VBitrate='1500k'; ABitrate='96k'  },
    @{ Id='CMP_LARGE_04'; Duration=600; Scale='1280:720'; VBitrate='2500k'; ABitrate='128k' },
    @{ Id='CMP_XL_05'; Duration=1200; Scale='1920:1080'; VBitrate='3500k'; ABitrate='128k' }
)

$videos = Get-ChildItem -Path $InputDir -File | Where-Object { $_.Extension -in '.mp4', '.mov', '.mkv', '.webm' }
if (-not $videos) {
    Write-Error "No input videos found in $InputDir"
    exit 1
}

foreach ($video in $videos) {
    foreach ($p in $profiles) {
        $name = [System.IO.Path]::GetFileNameWithoutExtension($video.Name)
        $outName = "{0}_{1}.mp4" -f $name, $p.Id
        $outPath = Join-Path $OutputDir $outName

        $cmd = @(
            '-y',
            '-i', $video.FullName,
            '-t', $p.Duration,
            '-vf', "scale=$($p.Scale)",
            '-c:v', 'libx264',
            '-b:v', $p.VBitrate,
            '-c:a', 'aac',
            '-b:a', $p.ABitrate,
            '-movflags', '+faststart',
            $outPath
        )

        Write-Host "Generating $outName"
        & $ffmpeg.Source @cmd | Out-Null
    }
}

Write-Host "Compression test set generated at: $OutputDir"
