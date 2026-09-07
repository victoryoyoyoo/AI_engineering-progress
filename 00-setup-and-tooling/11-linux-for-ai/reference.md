# Phase 0 Lesson 11: Linux for AI

在遠端GPU主機上會用到的15個核心指令,這堂課沒有獨立的程式碼檔案。

## 移動

```bash
pwd                         # 目前在哪
ls                          # 這裡有什麼
ls -la                      # 這裡有什麼,含隱藏檔跟詳細資訊
cd /path/to/dir             # 移過去
cd ~                        # 回家目錄
cd ..                       # 往上一層
```

## 檔案與目錄

```bash
mkdir my-project            # 建立目錄
mkdir -p a/b/c               # 一次建立巢狀目錄

cp file.txt backup.txt      # 複製檔案
cp -r src/ src-backup/      # 複製目錄(遞迴)

mv old.txt new.txt          # 重新命名
mv file.txt /tmp/           # 移動檔案

rm file.txt                 # 刪除檔案(沒有垃圾桶,直接消失)
rm -rf my-dir/              # 刪除整個目錄
```

`rm -rf`是永久的,沒有復原,下手前務必再三確認路徑。

## 讀檔案

```bash
cat file.txt                # 印出整個檔案
head -20 file.txt           # 前20行
tail -20 file.txt           # 後20行
tail -f log.txt             # 即時追蹤log(Ctrl+C停止)
less file.txt               # 分頁瀏覽(q離開)
```

## 搜尋

```bash
grep "error" training.log           # 找含有"error"的行
grep -r "learning_rate" .           # 搜尋目前目錄下所有檔案
grep -i "cuda" config.yaml          # 不分大小寫搜尋

find . -name "*.py"                 # 找出所有Python檔
find . -name "*.ckpt" -size +1G     # 找出超過1GB的checkpoint檔
```

## 權限

```bash
ls -l train.py
# -rwxr-xr-- 1 user group 2048 Mar 19 10:00 train.py
#  ^^^             owner權限:read write execute
#     ^^^          group權限:read execute
#        ^^        其他人:read only
```

常見修正:

```bash
chmod +x train.sh           # 讓腳本可執行
chmod 755 deploy.sh         # owner全權限,其他人可讀+執行
chmod 644 config.yaml       # owner可讀寫,其他人唯讀

chown user:group file.txt   # 改檔案擁有者(需要sudo)
```

## 套件管理(apt)

```bash
sudo apt update             # 先更新套件清單
sudo apt install -y htop    # 安裝套件(-y跳過確認)
sudo apt install -y build-essential  # C編譯器等,很多Python套件需要
sudo apt install -y tmux    # 終端多工器,斷線後session不會消失

apt list --installed        # 看已安裝什麼
sudo apt remove htop        # 移除
```

新的GPU主機常見的一次裝齊:

```bash
sudo apt update && sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    tmux \
    htop \
    unzip \
    python3-venv
```

## 使用者與sudo

```bash
whoami                      # 我是誰
sudo command                # 用root權限跑單一指令
sudo su                     # 變成root(exit回去,盡量少用)
```

## 行程與systemd

```bash
htop                        # 互動式行程檢視器(q離開)
ps aux | grep python        # 找Python行程
kill 12345                  # 溫和停止PID 12345
kill -9 12345               # 強制關閉
nvidia-smi                  # GPU行程跟記憶體用量

sudo systemctl start nginx          # 啟動服務
sudo systemctl stop nginx           # 停止
sudo systemctl restart nginx        # 重啟
sudo systemctl status nginx         # 檢查狀態
```
