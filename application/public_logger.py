import logging
from logging.handlers import RotatingFileHandler


logger = logging.getLogger("PublicLogger")
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(levelname)s %(asctime)s %(filename)s %(funcName)s line=%(lineno)d %(message)s',datefmt="%y-%m-%d %H:%M:%S")

file_handler = RotatingFileHandler("./log_files/log.txt", maxBytes=3*1024*1024, backupCount=10)#3M,10个日志文件
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console)

