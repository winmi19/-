git init
git config --global user.email "你的邮箱@example.com"
git config --global user.name "xuwenmin"
git add .

git commit -m "Initial commit: DINOv2 offline system"

git remote add origin git@gitlab.local:你的用户名/dinov2-offline.git
git remote add origin git@gitlab.local:xuwenmin/dinov2.git
git remote add origin ssh://git@172.31.179.162:2224/xuwenmin/dinov2.git

git remote add origin http://172.31.179.162:8929/4090/dinov2.git
git remote add origin http://172.31.179.162:8929/4090/dinov2.git

删除这个连不上的 HTTP 链接  git remote remove origin
git remote add origin ssh://git@172.31.179.162:2224/4090/dinov2.git换回 SSH 链接 (注意端口是 2224)
git remote add origin ssh://git@172.31.179.162:2224/4090/V-JEPA.git

git push -u origin（名字） main【注意看是main还是master】

git status 查看合并状态

git add README.md提交修改

git commit -m "Merge remote main branch with local changes"完成后提交

cd /mnt/d 去到d盘


ssh 4090 (注意添加config之后)   连接工作站

echo "您的公钥内容" >>
 ~/.ssh/authorized_keys设置密钥

 ssh -T git@github.com  检测是否连接成功 hi Winie!

apt-get update
apt-get install -y vim 下载vim

git submodule add https://github.com/facebookresearch/dinov2.git dinov2

git submodule add https://github.com/facebookresearch/jepa.git jepa

fuser -v /dev/nvidia0

nvidia-smi检查显卡