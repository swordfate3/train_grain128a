import os
import sys
import subprocess
from datetime import datetime
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GitAutoCommit:
    def __init__(self, repo_path, branch="main"):
        self.repo_path = Path(repo_path)
        self.branch = branch
        self.commit_message = f"Auto commit at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    def run_command(self, command):
        """执行shell命令并返回结果"""
        try:
            process = subprocess.run(
                command,
                cwd=self.repo_path,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            return process.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"命令执行失败: {e.cmd}")
            logger.error(f"错误信息: {e.stderr}")
            raise

    def check_repository(self):
        """检查仓库是否有效"""
        if not self.repo_path.exists():
            raise ValueError(f"仓库路径不存在: {self.repo_path}")

        if not (self.repo_path / ".git").is_dir():
            raise ValueError(f"不是有效的git仓库: {self.repo_path}")

    def check_and_switch_branch(self):
        """检查并切换到指定分支"""
        current_branch = self.run_command("git symbolic-ref --short HEAD")
        if current_branch != self.branch:
            logger.info(f"切换到 {self.branch} 分支")
            self.run_command(f"git checkout {self.branch}")

    def check_changes(self):
        """检查是否有变更需要提交"""
        status = self.run_command("git status --porcelain")
        return bool(status)

    def commit_and_push(self):
        """提交并推送代码"""
        try:
            # 添加所有变更
            self.run_command("git add .")

            # 提交变更
            self.run_command(f'git commit -m "{self.commit_message}"')

            # 推送到远程仓库
            self.run_command(f"git push origin {self.branch}")

            logger.info("代码提交成功！")
            return True
        except subprocess.CalledProcessError as e:
            logger.error("代码提交失败")
            return False

    def run(self):
        """执行完整的提交流程"""
        try:
            self.check_repository()
            self.check_and_switch_branch()

            if not self.check_changes():
                logger.info("没有需要提交的变更")
                return True

            logger.info("开始提交代码...")
            return self.commit_and_push()

        except Exception as e:
            logger.error(f"执行过程中出错: {str(e)}")
            return False


if __name__ == "__main__":
    # 配置信息
    REPO_PATH = "https://github.com/swordfate3/train_grain128a.git"  # 替换为您的仓库路径
    BRANCH = "grain"  # 替换为您要提交的分支

    # 创建并运行自动提交实例
    auto_commit = GitAutoCommit(REPO_PATH, BRANCH)
    success = auto_commit.run()

    # 设置退出码
    sys.exit(0 if success else 1)
