# Phase 0 Lesson 2: Git and Collaboration

Git工作流程參考,這堂課沒有獨立的程式碼檔案,重點是指令本身。

## 基本設定

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

## 日常工作流程

```bash
git status
git add file.py
git commit -m "Add perceptron implementation"
git push origin main
```

## 開分支做實驗

```bash
git checkout -b experiment/new-optimizer

# ... make changes, commit ...

git checkout main
git merge experiment/new-optimizer
```

## Fork課程repo

課程repo本身不能直接push,只有維護者有寫入權限,要先在GitHub上Fork一份:

```bash
git clone https://github.com/YOUR-USERNAME/ai-engineering-from-scratch.git
cd ai-engineering-from-scratch

git checkout -b my-progress
# work through lessons, commit your code
git push origin my-progress
```

## 這堂課只需要這幾個指令

| 指令 | 用途 |
|---|---|
| `git clone` | 取得課程repo |
| `git add` + `git commit` | 儲存工作進度 |
| `git push` | 備份到GitHub |
| `git checkout -b` | 開分支嘗試新東西,不動main |
| `git log --oneline` | 看做過什麼 |

不需要rebase、cherry-pick、submodule這些進階功能。
