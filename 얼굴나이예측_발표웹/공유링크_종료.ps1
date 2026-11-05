# 이번 발표용 임시 공유 연결만 종료합니다. 로컬 Streamlit 웹은 계속 실행됩니다.
$sharePidPath=Join-Path $PSScriptRoot '.runtime\share.pid'
if (Test-Path -LiteralPath $sharePidPath) {
    $shareProcessId=[int](Get-Content -LiteralPath $sharePidPath)
    $shareProcess=Get-CimInstance Win32_Process -Filter "ProcessId=$shareProcessId"
    if ($shareProcess -and $shareProcess.Name -eq 'cloudflared.exe' -and
        $shareProcess.CommandLine.Contains('http://127.0.0.1:8502')) {
        Stop-Process -Id $shareProcessId
        Write-Output '선생님께 보낸 임시 공유 링크를 종료했습니다.'
    } else {
        Write-Output '해당 공유 연결은 이미 종료되어 있습니다.'
    }
}
