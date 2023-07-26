import logging as log


def setupLog(level=log.INFO, simple=False):
    log.basicConfig(level=level)

    log.getLogger().handlers[0].setFormatter(
        log.Formatter(fmt="(%(asctime)s) [%(levelname)s] @%(filename)s # %(message)s", datefmt="%H:%M:%S")
    )
    if simple:
        return
    # Change colors depending on level
    log.addLevelName(log.DEBUG, "\033[1;34m%s\033[1;0m" % log.getLevelName(log.DEBUG))
    log.addLevelName(log.INFO, "\033[1;32m%s\033[1;0m" % log.getLevelName(log.INFO))
    log.addLevelName(log.WARNING, "\033[1;33m%s\033[1;0m" % log.getLevelName(log.WARNING))
    log.addLevelName(log.ERROR, "\033[1;31m%s\033[1;0m" % log.getLevelName(log.ERROR))
    log.addLevelName(log.CRITICAL, "\033[1;41m%s\033[1;0m" % log.getLevelName(log.CRITICAL))
