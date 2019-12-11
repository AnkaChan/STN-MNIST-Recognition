import logging
import sys

# logFileMode='a' or 'w', same as used in open(filename, *)

def configLogger(logFile, logFileMode='a', handlerStdoutLevel = logging.INFO, handlerFileLevel = logging.DEBUG, format=None):
    logger = logging.getLogger("logger")

    handlerStdout = logging.StreamHandler(sys.stdout)
    handlerFile = logging.FileHandler(filename=logFile, mode=logFileMode)

    logger.setLevel(logging.DEBUG)
    handlerStdout.setLevel(handlerStdoutLevel)
    handlerFile.setLevel(handlerFileLevel)

    if format is None:
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    else:
        formatter = logging.Formatter(format)

    handlerStdout.setFormatter(formatter)
    handlerFile.setFormatter(formatter)

    logger.addHandler(handlerStdout)
    logger.addHandler(handlerFile)

    return logger

if __name__ == '__main__':

    # logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    # rootLogger = logging.getLogger("TrainiLogger")
    #
    # fileHandler = logging.FileHandler("Log/" + "Test" + ".txt")
    # fileHandler.setFormatter(logFormatter)
    # rootLogger.addHandler(fileHandler)
    # rootLogger.addHandler(logging.StreamHandler(sys.stdout))
    #
    # rootLogger.info("INFO")
    # rootLogger.debug("DEBUG")

    # logger = logging.getLogger("logger")
    #
    # handlerStdout = logging.StreamHandler(sys.stdout)
    # handlerFile = logging.FileHandler(filename="Log/Test.log")
    #
    # logger.setLevel(logging.DEBUG)
    # handlerStdout.setLevel(logging.WARNING)
    # handlerFile.setLevel(logging.DEBUG)
    #
    # formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    # handlerStdout.setFormatter(formatter)
    # handlerFile.setFormatter(formatter)
    #
    # logger.addHandler(handlerStdout)
    # logger.addHandler(handlerFile)
    #
    # # print(handler1.level)
    # # print(handler2.level)
    # # print(logger.level)
    #
    # logger.debug('This is a customer debug message')
    # logger.info('This is an customer info message')
    # logger.warning('This is a customer warning message')
    # logger.error('This is an customer error message')
    # logger.critical('This is a customer critical message')

    logger = configLogger("Log/" + "Test" + ".txt", logFileMode='w')
    logger.info('info')
    logger.debug('debug', 123)