
import os

class DualLogger:
    """화면과 파일에 동시 출력하는 로거 클래스"""

    def __init__(self, log_path: str = None):
        self.log_path = log_path
        self.log_file = None
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.log_file = open(log_path, 'w', encoding='utf-8')

    def write(self, message: str):
        """메시지를 화면과 파일에 동시 출력"""
        print(message, end='')
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()

    def close(self):
        """파일 핸들 닫기"""
        if self.log_file:
            self.log_file.close()
            self.log_file = None

