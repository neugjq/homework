#!/bin/bash

# 构建网站
echo "Building website..."
npm run build

# 切换到 gh-pages 分支
cd ../
git checkout gh-pages

# 删除旧文件（保留 .git）
find . -maxdepth 1 ! -name '.git' ! -name '.' ! -name '..' -exec rm -rf {} +

# 复制新构建的文件
cp -r docs/build/* .

# 提交并推送
git add -A
git commit -m "Deploy website - $(date '+%Y-%m-%d %H:%M:%S')"
git push origin gh-pages

# 切换回 master 分支
git checkout master

echo "Deployment complete! Visit https://neugjq.github.io/homework/"
