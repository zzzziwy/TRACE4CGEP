
import logging


class Logger():
    def __init__(self, logname="info", loglevel=logging.DEBUG, loggername=None):
        # 创建一个logger
        self.logger = logging.getLogger(loggername)
        self.logger.setLevel(loglevel)
        # 创建一个handler，用于写入日志文件
        fh = logging.FileHandler(logname)
        fh.setLevel(loglevel)
        if not self.logger.handlers:
            # 或者使用如下语句判断
            # if not self.logger.hasHandlers():

            # 再创建一个handler，用于输出到控制台
            ch = logging.StreamHandler()
            ch.setLevel(loglevel)
            # 定义handler的输出格式
            # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            formatter = logging.Formatter('[%(levelname)s]%(asctime)s %(filename)s:%(lineno)d: %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            # 给logger添加handler
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

            self.logger.fatal("add handler")
        self.logger.fatal("set logger")

    def getlog(self):
        self.logger.fatal("get logger")
        return self.logger
