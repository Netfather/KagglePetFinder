# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         log
# Description:  此文件用于执行每一次的log保存
# Author:       Administrator
# Date:         2021/5/18
# -------------------------------------------------------------------------------

import logging
import os
import sys
import time


def get_logger(logdir, OutputOnConsole = True, log_initial = 'default', logfilename = 'run'):
    """
    本函数用于初始化日志必要： 定义日志文件名字，定义日志的级别，日志的输出格式，以及日志的流动方向
    :param logdir: 指定日志存储的位置
    :param OutputOnConsole: 指定日志是否需要输出到控制台
    :param log_initial: 指定日志初始化的头字符串 表示这个log主要用于记录什么
    :return:
    """

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logname = logfilename +f'-{time.strftime("%Y-%m-%d-%H-%M")}.log'  # Python time strftime() 返回以可读字符串表示的当地时间，格式由参数format决定
    log_file = os.path.join(logdir, logname)

    # create log
    logger = logging.getLogger(log_initial)     # log初始化
    logger.setLevel(logging.INFO)       # 设置log级别 设定为信息级别

    # Formatter 设置日志输出格式
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S') # 设置日志输出格式

    # StreamHandler 日志输出1 -> 到控制台
    # 如果需要输出到控制台，则指定输出到控制台
    if (OutputOnConsole):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler 日志输出2 -> 保存到文件log_file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger