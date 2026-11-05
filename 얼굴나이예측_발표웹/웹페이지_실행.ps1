# 발표 웹만 실행합니다. 학습 코드와 모델 파일은 수정하지 않습니다.
$webPython = Join-Path (Split-Path -Parent $PSScriptRoot) '전이학습\.venv\Scripts\python.exe'
$presentationApp = Join-Path $PSScriptRoot 'app.py'
$webRuntime = Join-Path $PSScriptRoot '.runtime'
New-Item -ItemType Directory -Path $webRuntime -Force | Out-Null
if (Get-NetTCPConnection -LocalPort 8502 -State Listen -ErrorAction SilentlyContinue) {
    Write-Output '8502 포트가 이미 실행 중입니다. http://127.0.0.1:8502 를 확인하세요.'
    return
}
$webServer = Start-Process -FilePath $webPython -ArgumentList @(
    '-m', 'streamlit', 'run', $presentationApp,
    '--server.headless=true', '--server.address=127.0.0.1', '--server.port=8502'
) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput (Join-Path $webRuntime 'output.log') `
  -RedirectStandardError (Join-Path $webRuntime 'error.log')
$webServer.Id | Set-Content -LiteralPath (Join-Path $webRuntime 'streamlit.pid')
Write-Output '발표 웹을 실행했습니다: http://127.0.0.1:8502'
