# 上传 GitHub

本目录已经是仓库根目录结构，建议仓库名使用 `ThermoResidual-RL`。
请创建一个新的空仓库，不要从包含现场数据的原工作目录直接推送历史提交。

1. 解压代码包，进入包含 `README.md` 的目录。
2. 在 GitHub 创建空仓库，选择公开可见性。不要另生成 README 或许可证，以免与包内文件冲突。
3. 在该目录执行以下命令，把最后的占位地址换成你实际创建的仓库地址：

```bash
git init
git add .
git status --short
git commit -m "Release ThermoResidual-RL reference implementation"
git branch -M main
git remote add origin <实际GitHub仓库地址>
git push -u origin main
```

上传前运行：

```bash
python reproduce.py --verify-only
python tools/audit_public_release.py
```

`.gitignore` 排除了输出、缓存、虚拟环境、常见私密文件与现场表格。
使用本代码包本身的新历史上传；不要把项目原始目录、整篇投稿包或个人账号文件一起提交。
隐藏的 `.github/` 和 `.gitignore` 也属于仓库，使用 `git add .` 会正常加入它们。

上传成功后可创建 `v1.0.0` release，并将本代码压缩包作为附件。
论文 Code availability 和回复信应填写真实仓库/release 链接；上传前不写成“已经公开”。
本包未代为登录或发布到任何 GitHub 账号，也没有填写虚构地址或 DOI。
