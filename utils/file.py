
import time
import uuid
def generate_unique_filename(suffix: str = "") -> str:
    """生成唯一文件名，格式：{timestamp}_{uuid}{suffix}"""
    return f"{int(time.time()*1000)}_{uuid.uuid4().hex}{suffix}"