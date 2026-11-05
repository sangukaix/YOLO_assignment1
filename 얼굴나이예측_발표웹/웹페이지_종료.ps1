# 처음 백그라운드로 실행한 이번 프로젝트의 서버를 종료합니다.
$agePidFile = Join-Path $PSScriptRoot '.runtime\streamlit.pid'
if (Test-Path -LiteralPath $agePidFile) {
    $ageServerId = [int](Get-Content -LiteralPath $agePidFile)
    $ageServer = Get-CimInstance Win32_Process -Filter "ProcessId=$ageServerId"
    $gpuPython = Join-Path (Split-Path -Parent $PSScriptRoot) '전이학습\.venv\Scripts\python.exe'
    $presentationApp = Join-Path $PSScriptRoot 'app.py'
    if ($ageServer -and $ageServer.CommandLine.Contains($gpuPython) -and
        $ageServer.CommandLine.Contains($presentationApp)) {
        # 가상환경의 실행기가 만든 자식 프로세스도 함께 종료합니다.
        & taskkill.exe /PID $ageServerId /T /F
    } else {
        Write-Output '처음 실행한 웹페이지 서버가 이미 종료되어 있습니다.'
    }
}
