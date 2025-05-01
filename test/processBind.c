// call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
// cl processBind.c /Fe:processBind.exe /utf-8 /W4 /MT /EHsc /link user32.lib kernel32.lib psapi.lib advapi32.lib
// powershell $p = get-process -name 'sguard64'; $p.processoraffinity=0xf000
#define UNICODE
#include <windows.h>
#include <tlhelp32.h>
#include <stdio.h>
#include <fcntl.h> // 
#include <io.h>
#define CONFIG_FILE L"bind.ini"
WCHAR exe_name[1000][MAX_PATH];
int mode[1000];
int count = 0;
// 全局常量，存储大小核掩码
DWORD_PTR g_BigCoreMask = 0;
DWORD_PTR g_LittleCoreMask = 0;
// 自动检测 CPU 大小核拓扑结构
void DetectCoreTypes() {
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX buffer = NULL;
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationAll, NULL, &length);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        wprintf(L"无法获取CPU信息，错误代码: %lu\n", GetLastError());
        return;
    }
    buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)malloc(length);
    if (!buffer) {
        wprintf(L"内存分配失败\n");
        return;
    }
    if (!GetLogicalProcessorInformationEx(RelationAll, buffer, &length)) {
        wprintf(L"无法获取CPU信息，错误代码: %lu\n", GetLastError());
        free(buffer);
        return;
    }
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX ptr = buffer;
    BYTE* end = (BYTE*)buffer + length;
    while ((BYTE*)ptr < end) {
        if (ptr->Relationship == RelationProcessorCore) {
            BOOL isEfficient = (ptr->Processor.EfficiencyClass > 0);
            for (WORD group = 0; group < ptr->Processor.GroupCount; group++) {
                KAFFINITY mask = ptr->Processor.GroupMask[group].Mask;
                if (isEfficient) {
                    g_BigCoreMask |= mask;
                } else {
                    g_LittleCoreMask |= mask;
                }
            }
        }
        ptr = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)((BYTE*)ptr + ptr->Size);
    }
    free(buffer);
    wprintf(L"检测到CPU拓扑:\n");
    wprintf(L"大核心掩码: 0x%llx\n", (unsigned long long)g_BigCoreMask);
    wprintf(L"小核心掩码: 0x%llx\n", (unsigned long long)g_LittleCoreMask);
}
DWORD find_process_id(const WCHAR *process_name) {
    PROCESSENTRY32 pe32;
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnapshot == INVALID_HANDLE_VALUE) return 0;
    pe32.dwSize = sizeof(PROCESSENTRY32);
    if (!Process32First(hSnapshot, &pe32)) {
        CloseHandle(hSnapshot);
        return 0;
    }
    do {
        if (_wcsicmp(pe32.szExeFile, process_name) == 0) {
            CloseHandle(hSnapshot);
            return pe32.th32ProcessID;
        }
    } while (Process32Next(hSnapshot, &pe32));
    CloseHandle(hSnapshot);
    return 0;
}
void trim_newline(WCHAR *str) {
    size_t len = wcslen(str);
    while (len > 0 && (str[len-1] == L'\n' || str[len-1] == L'\r')) {
        str[--len] = L'\0';
    }
}
const WCHAR* get_last_error_message() {
    static WCHAR msg[256];
    FormatMessageW(  // 使用宽字符版本
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        GetLastError(),
        0,
        msg,
        sizeof(msg)/sizeof(WCHAR),  // 注意缓冲区大小以字符计
        NULL
    );
    return msg;
}
void bind_if_found(const WCHAR *target_exe, int target_mode) {
    DWORD_PTR mask = (target_mode == 1) ? g_LittleCoreMask : g_BigCoreMask;
    const WCHAR *core_label = (target_mode == 1) ? L"小核心" : L"大核心";
    DWORD pid = find_process_id(target_exe);
    if (pid == 0) return;
    HANDLE hProcess = OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION, FALSE, pid);
    if (!hProcess) {
        wprintf(L"无法打开 %ls（PID=%lu），错误: %ls\n", target_exe, pid, get_last_error_message());
        return;
    }
    if (SetProcessAffinityMask(hProcess, mask)) {
        wprintf(L"绑定成功：%ls -> %ls\n", target_exe, core_label);
    } else {
        wprintf(L"绑定失败：%ls，错误代码: %lu\n", target_exe, GetLastError());
    }
    CloseHandle(hProcess);
}
// 用于监听前台窗口事件（轻量监听，无需全盘扫描）
void CALLBACK win_event_proc(
    HWINEVENTHOOK hWinEventHook, DWORD event, HWND hwnd, LONG idObject, LONG idChild,
    DWORD dwEventThread, DWORD dwmsEventTime)
{   
    WCHAR exe_path[MAX_PATH];
    DWORD pid = 0;
    GetWindowThreadProcessId(hwnd, &pid);
    if (pid == 0) return;
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (hProcess) {
        DWORD len = MAX_PATH;
        if (QueryFullProcessImageNameW(hProcess, 0, exe_path, &len)) {
            const WCHAR *base_name = wcsrchr(exe_path, '\\');
            if (base_name) base_name++; else base_name = exe_path;
            for (int i = 0; i < count; i++) {
                if (_wcsicmp(base_name, exe_name[i]) == 0) {
                    bind_if_found(base_name, mode[i]);
                    break;
                }
            }
        }
        CloseHandle(hProcess);
    }
}
void scan_and_bind_all() {
    for (int i = 0; i < count; i++)
    {
        bind_if_found(exe_name[i], mode[i]);
    }
}
void readIniFile() {
    FILE* file = NULL;
    errno_t err = _wfopen_s(&file, CONFIG_FILE, L"r, ccs=UTF-8");  // 使用安全版本
    if (err != 0 || !file) {
        WCHAR userPath[MAX_PATH];
        swprintf_s(userPath, MAX_PATH, L"C:\\Users\\%s\\Desktop\\c\\bind.ini", _wgetenv(L"USERNAME"));
        err = _wfopen_s(&file, userPath, L"r, ccs=UTF-8");
        if (err != 0 || !file) {
            wprintf(L"未找到配置文件：%ls\n", CONFIG_FILE);
            return;
        }
    }
    WCHAR line[256];
    wprintf(L"读取配置：\n");
    while (fgetws(line, sizeof(line)/sizeof(WCHAR), file)) {  // 使用fgetws代替fgets
        trim_newline(line);
        if (line[0] == L'#' || line[0] == L'\0') continue; // 注意使用宽字符常量L''
        WCHAR *last_space = wcsrchr(line, L' ');  // 使用wcsrchr代替strrchr
        if (!last_space) {
            wprintf(L"格式错误：%ls\n", line);
            continue;
        }
        *last_space = L'\0'; // 分割为两个字符串
        mode[count] = _wtoi(last_space + 1);  // 使用_wtoi代替atoi
        wcsncpy_s(exe_name[count], MAX_PATH, line, _TRUNCATE);
        exe_name[count++][MAX_PATH - 1] = L'\0';
        wprintf(L"程序文件名：\"%ls\" 模式：%d\n", exe_name[count-1], mode[count-1]);
    }
    fclose(file);
}
BOOL EnableDebugPrivilege() {
    HANDLE hToken;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES, &hToken))
        return FALSE;
    TOKEN_PRIVILEGES tkp;
    LookupPrivilegeValue(NULL, SE_DEBUG_NAME, &tkp.Privileges[0].Luid);
    tkp.PrivilegeCount = 1;
    tkp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    BOOL success = AdjustTokenPrivileges(hToken, FALSE, &tkp, 0, NULL, 0);
    CloseHandle(hToken);
    return success;
}

int main() {
    // HWND hwnd = GetConsoleWindow();
    // ShowWindow(hwnd, SW_HIDE); // SW_HIDE = 0, SW_SHOW = 5 不显示窗口
    _setmode(_fileno(stdout), _O_U16TEXT);  // 需要 <fcntl.h> 和 <io.h>
    EnableDebugPrivilege();
    DetectCoreTypes();
    if (g_BigCoreMask == 0 && g_LittleCoreMask == 0) {
        wprintf(L"使用默认核心掩码\n");
        g_BigCoreMask = 0xFFF;     // 核心0-11
        g_LittleCoreMask = 0xF000; // 核心12-15
    }
    SetProcessAffinityMask(GetCurrentProcess(), g_LittleCoreMask);
    readIniFile();
    scan_and_bind_all();
    HWINEVENTHOOK hHook = SetWinEventHook(
        EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_FOREGROUND,
        NULL, win_event_proc, 0, 0, WINEVENT_OUTOFCONTEXT);
    if (!hHook) {
        wprintf(L"无法设置钩子事件，错误代码: %lu\n", GetLastError());
        return 1;
    }
    wprintf(L"监控程序已启动，等待配置中的目标进程启动...\n");
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    UnhookWinEvent(hHook);
    return 0;
}
