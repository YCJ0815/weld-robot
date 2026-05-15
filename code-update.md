# 代码同步说明

本文说明本地代码与服务器端 Git 仓库的同步方式。

当前远端仓库：

```bash
origin root@connect.bjb1.seetacloud.com:/root/git/weld-robot.git
```

当前默认分支：

```bash
main
```

## 本地修改后推送到服务器端

在本地项目根目录执行：

```bash
git status
git add .
git commit -m "说明本次修改内容"
git push origin main
```

```
# 服务器
cd /root/weld-robot
git pull ~/git/weld-robot.git main
```

说明：

- `git status` 用于确认哪些文件被修改。
- `git add .` 会暂存当前目录下所有修改。
- `git commit -m` 用一句话记录本次修改。
- `git push origin main` 将本地提交推送到服务器端仓库。

## 服务器端目录架构修改后拉取到本地

如果服务器端已经提交并推送了目录结构或文件修改，在本地执行：

```bash
git status
git pull origin main
```

建议先运行 `git status`，确认本地没有未提交修改。若本地也有修改，先提交再拉取：

```bash
git add .
git commit -m "保存本地修改"
git pull origin main
```

## 出现冲突时

如果 `git pull` 后提示冲突：

1. 打开冲突文件，保留需要的内容。
2. 修改完成后执行：

```bash
git add .
git commit -m "解决同步冲突"
git push origin main
```

## 常用检查命令

```bash
git status
git log --oneline -5
git remote -v
```

## 实例被占用
在mac上输入
```
