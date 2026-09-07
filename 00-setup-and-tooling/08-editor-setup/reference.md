# Phase 0 Lesson 8: Editor Setup

VS Code設定參考,這堂課沒有獨立的程式碼檔案,重點是設定本身。

## 安裝VS Code

從 [code.visualstudio.com](https://code.visualstudio.com/) 下載,終端機驗證:

```bash
code --version
```

## 必裝擴充套件

```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension eamodio.gitlens
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension ms-python.debugpy
code --install-extension ms-python.black-formatter
code --install-extension charliermarsh.ruff
```

| 擴充套件 | 用途 |
|---|---|
| Python | 語言支援、虛擬環境偵測、執行/除錯 |
| Pylance | 快速型別檢查、自動完成、import解析 |
| Jupyter | 在VS Code裡直接跑notebook,含變數檢視器 |
| GitLens | 看誰改了什麼,行內git blame |
| Remote SSH | 把遠端GPU主機的資料夾當本機資料夾一樣打開 |
| Debugpy | Python逐步除錯 |
| Black Formatter | 存檔自動格式化,風格一致 |
| Ruff | 快速linting,抓常見錯誤 |

## 設定(settings.json)

```jsonc
{
    "python.analysis.typeCheckingMode": "basic",
    "editor.formatOnSave": true,
    "editor.rulers": [88, 120],
    "notebook.output.scrolling": true,
    "files.autoSave": "afterDelay"
}
```

## 終端機整合

```jsonc
{
    "terminal.integrated.defaultProfile.osx": "zsh",
    "terminal.integrated.defaultProfile.linux": "bash",
    "terminal.integrated.fontSize": 13,
    "terminal.integrated.scrollback": 10000
}
```
