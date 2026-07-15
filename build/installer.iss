; Inno Setup 安装脚本 — 贾维斯语音助手
; 需要 Inno Setup 6+ (https://jrsoftware.org/isinfo.php)
; 编译: iscc installer.iss

#define MyAppName "贾维斯语音助手"
#define MyAppNameShort "Jarvis"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "Jarvis Team"
#define MyAppURL "https://github.com/jarvis-assistant"
#define MyAppExeName "Jarvis.exe"

[Setup]
AppId={{B8F4A3D2-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppNameShort}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=..\dist_installer
OutputBaseFilename=JarvisSetup_v{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64compatible
ChangesEnvironment=yes

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式："
Name: "autostart"; Description: "开机自动启动"; GroupDescription: "启动选项："
Name: "ollama"; Description: "安装 Ollama（本地 LLM 所需，约 800MB）"; GroupDescription: "依赖安装："

[Files]
; 主程序文件
Source: "..\dist\Jarvis\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; 模型目录（可选，如果存在）
Source: "..\models\*"; DestDir: "{localappdata}\{#MyAppNameShort}\models"; Flags: ignoreversion recursesubdirs createallsubdirs; Check: DirExists(ExpandConstant('..\models'))

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{userstartup}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: autostart

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent; Check: not WizardSilent

[UninstallRun]
Filename: "{app}\{#MyAppExeName}"; Parameters: "--uninstall"; Flags: runhidden

[Code]
function ShouldInstallOllama: Boolean;
begin
  Result := IsTaskSelected('ollama');
end;

procedure InstallOllama;
var
  ResultCode: Integer;
  DownloadPage: TDownloadWizardPage;
begin
  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing), SetupMessage(msgPreparingDesc), nil);
  DownloadPage.Show;
  DownloadPage.Clear;
  DownloadPage.Add('https://ollama.com/download/OllamaSetup.exe', 'OllamaSetup.exe', '');
  try
    DownloadPage.Download;
    if Exec(ExpandConstant('{tmp}\OllamaSetup.exe'), '', '', SW_SHOW, ewWaitUntilTerminated, ResultCode) then
      MsgBox('Ollama 安装完成！', mbInformation, MB_OK)
    else
      MsgBox('Ollama 安装失败，请手动下载安装：https://ollama.com', mbError, MB_OK);
  finally
    DownloadPage.Hide;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if ShouldInstallOllama then
      InstallOllama;
  end;
end;
